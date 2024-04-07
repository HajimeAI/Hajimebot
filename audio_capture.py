import os
import sys
import time
import json
import websockets
import asyncio
import numpy as np
import torch

from configs import logger, WS_MAX_SIZE
from whisper.audio_recorder import AudioRecorder, pack_audio_chunks, dump_wave_file
from communication.models import *

_local_ws = None
async def send_message_to_local(message):
    global _local_ws
    if not _local_ws:
        return
    
    await _local_ws.send(message)

async def record_pause(req: WsBase, audio_recorder: AudioRecorder):    
    audio_recorder.pause()

async def record_resume(req: WsBase, audio_recorder: AudioRecorder):    
    audio_recorder.resume()

action_list = [
    (record_pause, WsBase),
    (record_resume, WsBase),
]
actions_map = {}
for x in action_list:
    actions_map[x[0].__name__] = x

async def start_local_ws_connection(audio_recorder):
    async for websocket in websockets.connect(
        "ws://127.0.0.1:5001",
        max_size=WS_MAX_SIZE,
    ):
        logger.info('websocket connected.')
        global _local_ws
        _local_ws = websocket
        try:
            async for message in websocket: 
                logger.info(f"Received local ws message: {message}")
                event = json.loads(message)
                reqBase = WsBase.model_validate(event)
                if reqBase.action not in actions_map:
                    logger.error(f'Invalid action: {reqBase.action}')
                    continue

                func, req = actions_map[reqBase.action]
                await func(req, audio_recorder)

        except websockets.ConnectionClosed as e:
            logger.info(f'websocket error: {e}, reconnect...')
            audio_recorder.resume()
            continue

def int_to_float(sound):
    _sound = np.copy(sound)
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
        _sound *= 1 / abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32

async def capture_sentences(audio_recorder: AudioRecorder):
    time.sleep(2)

    silero_vad_dir = os.path.join(os.path.dirname(__file__), "silero-vad")
    model, utils = torch.hub.load(
        repo_or_dir=silero_vad_dir,
        model='silero_vad',
        source='local',
        force_reload=True,
        onnx=False
    )

    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks
    ) = utils

    while audio_recorder.is_active():
        sentences = audio_recorder.fetch_sentences()
        if len(sentences) == 0:
            await asyncio.sleep(0.1)
            continue

        sent = False
        for chunks in sentences:
            raw_data = pack_audio_chunks(chunks)

            start_tm = time.time()
            newsound = np.frombuffer(raw_data, np.int16)
            audio_float32 = int_to_float(newsound)
            time_stamps = get_speech_timestamps(
                audio_float32, 
                model,
                min_speech_duration_ms=300,  # min speech duration in ms
                min_silence_duration_ms=600,  # min silence duration
                speech_pad_ms=200,  # spech pad ms
            )

            elasped = time.time() - start_tm
            if len(time_stamps) > 0:
                logger.info(f"silero VAD has detected a possible speech, elasped: {elasped:.02f}s")
                dump_wave_file(raw_data, sub_dir='record/')
            else:
                logger.info(f"silero VAD has detected a noise, elasped: {elasped:.02f}s")
                continue
            
            await send_message_to_local(raw_data)
            sent = True
        
        if sent:
            # pause until recv record_resume message
            audio_recorder.pause()

async def main(audio_recorder: AudioRecorder):
    tasks = [
        start_local_ws_connection(audio_recorder),
        capture_sentences(audio_recorder),
    ]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    audio_recorder = AudioRecorder()

    # wait for 1 second to avoid any buffered noise
    time.sleep(1)
    audio_recorder.start()

    try:
        asyncio.run(main(audio_recorder))
    except KeyboardInterrupt:
        sys.stdout.write('\n')
        sys.stdout.flush()
        logger.info('>>> Stop audio recording')

    audio_recorder.pause()
    del audio_recorder
