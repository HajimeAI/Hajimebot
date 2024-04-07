import sys
import threading
from array import array
import struct
from io import BytesIO
import time
import datetime as dt
from typing import List
import pyaudio
import wave
import webrtcvad
import collections

from configs import logger, SPEECH_DIR
import whisper.utils as utils

def pack_audio_chunks(chunks: List[bytes]) -> bytes:
    raw_data = array('h')
    for chunk in chunks:
        raw_data.extend(array('h', chunk))
    
    raw_data = normalize(raw_data)
    with BytesIO() as buf:
        for val in raw_data:
            buf.write(struct.pack('<h', val))

        return buf.getvalue()

# normalize the captured samples
def normalize(snd_data: array):
    MAXIMUM = 32767
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r

def dump_wave_file(raw_data: bytes, sub_dir=''):
    file_path = dt.datetime.now().strftime(f'{SPEECH_DIR}/{sub_dir}%Y%m%d_%H%M%S.wav')
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(AudioRecorder.CHANNELS)
        wf.setsampwidth(AudioRecorder.WIDTH)
        wf.setframerate(AudioRecorder.RATE)
        wf.writeframes(raw_data)

def is_speech_complete(pcmf32: List[float]) -> bool:
    return utils.vad_simple(
        pcmf32=pcmf32, 
        sample_rate=AudioRecorder.RATE, 
        last_ms=AudioRecorder.SIMPLE_VAD_AUDIO_REAR_MS, 
        vad_thold=AudioRecorder.SIMPLE_VAD_THOLD, 
        freq_thold=AudioRecorder.SIMPLE_VAD_FREQ_THOLD, 
    )

class AudioRecorder():
    # sample width
    WIDTH = 2
    # channels
    CHANNELS = 1
    # samples per second
    RATE = 16000
    # chunk duration(ms), support 10/20/30 ms
    CHUNK_DURATION_MS = 30
    # samples in chunk
    CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
    # chunks in voice starting detection window
    NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)
    # chunks in voice ending detection window
    NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2
    # max recording seconds
    MAX_RECORDING_SECONDS = 60
    # move forward some chunks from voice found point
    START_OFFSET = max(20, NUM_WINDOW_CHUNKS)

    SIMPLE_VAD_AUDIO_TOTAL_MS = 3000
    SIMPLE_VAD_AUDIO_REAR_MS = 1500
    SIMPLE_VAD_AUDIO_CHUNK_SIZE = int(RATE * SIMPLE_VAD_AUDIO_TOTAL_MS / 1000)
    SIMPLE_VAD_THOLD = 0.3
    SIMPLE_VAD_FREQ_THOLD = 100.0

    def __init__(
        self, 
        device_id: int = -1
    ):
        def stream_callback(in_data, frame_count, time_info, status):
            self.on_data_received(in_data)
            return (in_data, pyaudio.paContinue)

        self._pa = pyaudio.PyAudio()

        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            print((i, dev['name'], dev['maxInputChannels']))

        self._stream = self._pa.open(
            rate=AudioRecorder.RATE,
            channels=AudioRecorder.CHANNELS,
            format=self._pa.get_format_from_width(AudioRecorder.WIDTH),
            input=True,
            input_device_index=(device_id if device_id >= 0 else None),
            frames_per_buffer=AudioRecorder.CHUNK_SIZE,
            stream_callback=stream_callback
        )
        self._vad = webrtcvad.Vad(3)

        self._sentences = []
        self._sentence_lock = threading.Lock()
        self._running = False

        self._pcmf32_buffer = collections.deque(maxlen=AudioRecorder.SIMPLE_VAD_AUDIO_CHUNK_SIZE)

        self.reset()

    def reset(self):
        self._triggered = False

        self._ring_buffer_flags = [0] * AudioRecorder.NUM_WINDOW_CHUNKS
        self._ring_buffer_index = 0

        self._ring_buffer_flags_end = [0] * AudioRecorder.NUM_WINDOW_CHUNKS_END
        self._ring_buffer_index_end = 0

        self._chunks = []
        self._start_index = 0
        self._start_time = 0.0

        self._pcmf32_buffer.clear()

        logger.info("Start recording: ")

    def on_data_received(self, in_data: bytes):
        if not self._running:
            return
        
        self._chunks.append(in_data)
        int16_arr = array('h', in_data)
        for x in int16_arr:
            self._pcmf32_buffer.append(float(x) / 32768.0)

        active = self._vad.is_speech(in_data, AudioRecorder.RATE)
        sys.stdout.write('1' if active else '_')

        self._ring_buffer_flags[self._ring_buffer_index] = 1 if active else 0
        self._ring_buffer_index += 1
        if self._ring_buffer_index >= AudioRecorder.NUM_WINDOW_CHUNKS:
            self._ring_buffer_index = 0

        self._ring_buffer_flags_end[self._ring_buffer_index_end] = 1 if active else 0
        self._ring_buffer_index_end += 1
        if self._ring_buffer_index_end >= AudioRecorder.NUM_WINDOW_CHUNKS_END:
            self._ring_buffer_index_end = 0

        # voice starting detection
        got_a_sentence = False
        if not self._triggered:
            self._start_index += 1
            num_voiced = sum(self._ring_buffer_flags)
            if num_voiced > 0.8 * AudioRecorder.NUM_WINDOW_CHUNKS:
                sys.stdout.write(' Open ')
                self._triggered = True
                self._start_index -= AudioRecorder.START_OFFSET
                if self._start_index < 0:
                    self._start_index = 0
                self._chunks = self._chunks[self._start_index : ]
                self._start_time = time.time()

        # voice ending detection
        else:
            simple_vad_complete = is_speech_complete(list(self._pcmf32_buffer))

            elasped_secs = time.time() - self._start_time
            num_unvoiced = AudioRecorder.NUM_WINDOW_CHUNKS_END - sum(self._ring_buffer_flags_end)
            webrtcvad_complete = (num_unvoiced > 0.90 * AudioRecorder.NUM_WINDOW_CHUNKS_END
                or elasped_secs > AudioRecorder.MAX_RECORDING_SECONDS)
            if (simple_vad_complete or webrtcvad_complete):
                # got a sentence
                sys.stdout.write(f' Close[{0 if simple_vad_complete else 1}] \n')
                got_a_sentence = True

        sys.stdout.flush()
        if got_a_sentence:
            with self._sentence_lock:
                self._sentences.append(self._chunks)
            
            self.reset()
    
    def fetch_sentences(self) -> List:
        result = []
        with self._sentence_lock:
            result = self._sentences
            self._sentences = []

        return result

    def start(self):
        self._running = True

    def pause(self, should_stop_stream=False):
        if not self._running:
            return
        
        if should_stop_stream:
            self._stream.stop_stream()
        self._running = False

        logger.info('audio record paused.')

    def resume(self, should_start_stream=False):
        if self._running:
            return
        
        self.reset()
        if should_start_stream:
            self._stream.start_stream()
        self._running = True

        logger.info('audio record resumed.')

    def is_active(self) -> bool:
        return self._stream.is_active()

    def __del__(self):
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()
