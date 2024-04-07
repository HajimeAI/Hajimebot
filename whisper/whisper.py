import time
import wave
import zhconv
from io import BytesIO
import typing
from typing import List, Dict, Tuple
import re
import math
from enum import Enum

from configs import (
    logger, 
    PROMPT_TEMPLATES, 
    WHISPER_PROMPT_LEN_DELTA, 
    WHISPER_PROMPT_SIM_THOLD,
    WHISPER_COMMAND_SIM_THOLD,
)

from .whisper_types import *
from .bindings import *
from .utils import *
from .audio_recorder import AudioRecorder
from model_api.tts import play_const_audio

def cb_log_disable(ggml_log_level: c_int, text: c_void_p, user_data: c_void_p):
    pass

POINTER_USER_DATA = POINTER(whisper_print_user_data)
def whisper_print_segment_callback(ctx: c_void_p, state: c_void_p, n_new: c_int, user_data: POINTER_USER_DATA):
    params = user_data.contents.params
    pcmf32s = user_data.contents.pcmf32s

    n_segments = whisper_full_n_segments(ctx)

    speaker = ""
    t0 = 0
    t1 = 0
    # print the last n_new segments
    s0 = n_segments - n_new
    if s0 == 0:
        print("")
    
    out_text = ""
    for i in range(s0, n_segments):
        if not params.no_timestamps or params.diarize:
            t0 = whisper_full_get_segment_t0(ctx, i)
            t1 = whisper_full_get_segment_t1(ctx, i)
        
        if not params.no_timestamps:
            out_text += f"[{to_timestamp(t0)} --> {to_timestamp(t1)}]  "

        if params.diarize and pcmf32s.size() == 2:
            speaker = estimate_diarization_speaker(pcmf32s, t0, t1);

        if params.print_colors:
            tokens = whisper_full_n_tokens(ctx, i)
            for j in range(tokens):
                if not params.print_special:
                    id = whisper_full_get_token_id(ctx, i, j)
                    if id >= whisper_token_eot(ctx):
                        continue

                text = whisper_full_get_token_text(ctx, i, j)
                p = whisper_full_get_token_p(ctx, i, j)

                col = max(0, min(len(k_colors) - 1, int((p ** 3) * len(k_colors))))

                out_text += f"{speaker}{k_colors[col]}{text}\033[0m"
        else:
            text = whisper_full_get_segment_text(ctx, i)
            out_text += f"{speaker}{text}"

        if params.tinydiarize:
            if whisper_full_get_segment_speaker_turn_next(ctx, i):
                out_text += f'{params.tdrz_speaker_turn}'

        # with timestamps or speakers: each segment on new line
        if not params.no_timestamps or params.diarize:
            print(out_text)
            out_text = ""

class CmdStatus(Enum):
    NO_PROMPT = 0,
    HAVE_PROMPT = 1,
    CONVERSATION = 2,

class Whisper():
    def __init__(self, model_path: str, **kwargs):
        self._params = WhisperParams()
        self._params.model = model_path

        self.update_params(**kwargs)

        if self._params.diarize and self._params.tinydiarize:
            raise WhisperException("error: cannot use both --diarize and --tinydiarize")
        
        if self._params.no_prints:
            whisper_log_set(GGML_LOG_CALLBACK(cb_log_disable), c_void_p())
        
        cparams = whisper_context_default_params()
        cparams.use_gpu = self._params.use_gpu

        self.context = whisper_init_from_file_with_params(model_path, cparams)
        if not self.context:
            raise WhisperException("error: failed to initialize whisper context")

        logger.info(f'whisper_init_from_file, context: {hex(self.context)}')

        logger.info(f'system_info: n_threads = {self._params.n_threads * self._params.n_processors} | {whisper_print_system_info()}')

        self._blank_pattern1 = re.compile('\[.*?\]', re.M|re.S)
        self._blank_pattern2 = re.compile('\(.*?\)', re.M|re.S)

        self.init_commands()

    def check_lang_setting(self):
        if self._params.language != 'auto' and whisper_lang_id(self._params.language) == -1:
            raise WhisperException(f"error: unknown language '{self._params.language}'")
        if not whisper_is_multilingual(self.context):
            if self._params.language != 'en' or self._params.translate:
                self._params.language = 'en'
                self._params.translate = False
                logger.warn(f'WARNING: model is not multilingual, ignoring language and translation options')
        if self._params.detect_language:
            self._params.language = 'auto'

    def update_params(self, *args, **kwargs):
        # print(f'update_params: {kwargs}')
        for k, v in kwargs.items():
            setattr(self._params, k, v)

    def transcribe_wav(
            self, 
            file_path: str, 
            lang: str, 
            **kwargs
        ) -> Tuple[str, str, float, float, float]:
        self._params.fname_inp = file_path
        self._params.language = lang

        self.update_params(**kwargs)

        self.check_lang_setting()
        
        pcmf32, pcmf32s = self._read_wav(self._params.fname_inp, self._params.diarize)

        logger.info(f"""
            processing '{self._params.fname_inp}' ({len(pcmf32)} samples, {len(pcmf32) / WHISPER_SAMPLE_RATE:.1f} sec), 
            {self._params.n_threads} threads, {self._params.n_processors} processors, lang = {self._params.language}, 
            task = {'translate' if self._params.translate else 'transcribe'}, {'tdrz = 1, ' if self._params.tinydiarize else ''}timestamps = {0 if self._params.no_timestamps else 1}
        """)

        return self.transcribe(pcmf32, pcmf32s)
    
    def handle_command_workflow(
        self, 
        raw_data: bytes,
        sample_width: int,
        lang: str, 
        **kwargs
    ) -> Tuple[str, str, float, float]:
        self._params.language = lang
        
        self.update_params(**kwargs)

        self.check_lang_setting()

        pcmf32 = []
        pcmf32s = []
        nframes = len(raw_data) // sample_width
        with BytesIO(raw_data) as bio:
            frames = [int.from_bytes(bio.read(sample_width), byteorder='little', signed=True) for _ in range(nframes)]
            pcmf32 = [ float(frames[x]) / 32768.0 for x in range(nframes) ]
        
        org_cmd_status = self._cmd_stats
        pcmf32_cmd = pcmf32
        if org_cmd_status == CmdStatus.HAVE_PROMPT:
            pcmf32_cmd = self._pcmf32_prompt + pcmf32

        self.process_commands(pcmf32_cmd)
        if org_cmd_status == CmdStatus.CONVERSATION and org_cmd_status == self._cmd_stats:
            return self.transcribe(pcmf32, pcmf32s)

    def transcribe_raw(
            self, 
            raw_data: bytes,
            sample_width: int,
            lang: str, 
            **kwargs
        ) -> Tuple[str, str, float, float]:
        self._params.language = lang
        
        self.update_params(**kwargs)

        self.check_lang_setting()

        pcmf32 = []
        pcmf32s = []
        nframes = len(raw_data) // sample_width
        with BytesIO(raw_data) as bio:
            frames = [int.from_bytes(bio.read(sample_width), byteorder='little', signed=True) for _ in range(nframes)]
            pcmf32 = [ float(frames[x]) / 32768.0 for x in range(nframes) ]

        return self.transcribe(pcmf32, pcmf32s)

    def transcribe(self, pcmf32: List[float], pcmf32s: List[List[float]] = []) -> Tuple[str, str, float, float]:
        start_tm = time.perf_counter()

        wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        wparams.strategy = WHISPER_SAMPLING_BEAM_SEARCH if self._params.beam_size > 1 else WHISPER_SAMPLING_GREEDY

        wparams.print_realtime   = False
        wparams.print_progress   = self._params.print_progress
        wparams.print_timestamps = not self._params.no_timestamps
        wparams.print_special    = self._params.print_special
        wparams.translate        = self._params.translate
        wparams.language         = c_char_p(self._params.language.encode('utf-8'))
        wparams.detect_language  = self._params.detect_language
        wparams.n_threads        = self._params.n_threads
        wparams.n_max_text_ctx   = self._params.max_context if self._params.max_context >= 0 else wparams.n_max_text_ctx
        wparams.offset_ms        = self._params.offset_t_ms
        wparams.duration_ms      = self._params.duration_ms

        wparams.token_timestamps = self._params.output_wts or self._params.output_jsn_full or self._params.max_len > 0
        wparams.thold_pt         = self._params.word_thold
        wparams.max_len          = 60 if self._params.output_wts and self._params.max_len == 0 else self._params.max_len
        wparams.split_on_word    = self._params.split_on_word
        wparams.audio_ctx        = self._params.audio_ctx

        wparams.speed_up         = self._params.speed_up
        wparams.debug_mode       = self._params.debug_mode

        wparams.tdrz_enable      = self._params.tinydiarize

        wparams.initial_prompt   = c_char_p(self._params.prompt.encode('utf-8'))

        wparams.greedy.best_of        = self._params.best_of
        wparams.beam_search.beam_size = self._params.beam_size

        wparams.temperature_inc  = 0.0 if self._params.no_fallback else wparams.temperature_inc
        wparams.entropy_thold    = self._params.entropy_thold
        wparams.logprob_thold    = self._params.logprob_thold

        wparams.no_timestamps    = self._params.no_timestamps

        if not wparams.print_realtime:
            wparams.new_segment_callback = WHISPER_NEW_SEQ_CALLBACK(whisper_print_segment_callback)

            user_data = whisper_print_user_data(
                params = py_object(self._params), 
                pcmf32s = py_object(pcmf32s), 
                progress_prev = 0
            )
            wparams.new_segment_callback_user_data = pointer(user_data)

        code = whisper_full_parallel(self.context, wparams, pcmf32, self._params.n_processors)
        if code != 0:
            raise WhisperException(f'whisper_full_parallel failed to process audio')
        
        lang_id = whisper_full_lang_id(self.context)
        lang = whisper_lang_str(lang_id)
        
        result, prob = self._output_txt(pcmf32s)

        elapsed = time.perf_counter() - start_tm

        return lang, result, prob, elapsed

    # The sample rate of the audio must be equal to COMMON_SAMPLE_RATE
    # If stereo flag is set and the audio has 2 channels, the pcmf32s will contain 2 channel PCM
    def _read_wav(self, fname, stereo) -> Tuple[List[float], List[List[float]]]:
        pcmf32 = []
        pcmf32s = []
        with wave.open(fname, "rb") as wavfile:
            params = wavfile.getparams()
            if params.nchannels != 1 and params.nchannels != 2:
                raise WhisperException(f"WAV file '{fname}' must be mono or stereo")
            if stereo and params.nchannels != 2:
                raise WhisperException(f"WAV file '{fname}' must be stereo for diarization")
            if params.framerate != WHISPER_SAMPLE_RATE:
                raise WhisperException(f"WAV file '{fname}' must be {WHISPER_SAMPLE_RATE/1000} kHz")
            if params.sampwidth != 2:
                raise WhisperException(f"WAV file '{fname}' must be 16-bit")
            
            bs = wavfile.readframes(params.nframes)
            logger.info(f'{params}, readframes: {len(bs)}')

            with BytesIO(bs) as bio:
                frames = [int.from_bytes(bio.read(2), byteorder='little', signed=True) for _ in range(params.nframes * params.nchannels)]           
                if params.nchannels == 1:
                    pcmf32 = [ float(frames[x]) / 32768.0 for x in range(params.nframes) ]
                else:
                    pcmf32 = [ float(frames[2*x] + frames[2*x+1]) / 65536.0 for x in range(params.nframes) ]

                if stereo:
                    pcmf32s.append([ float(frames[2*x]) / 32768.0 for x in range(params.nframes) ])
                    pcmf32s.append([ float(frames[2*x+1]) / 32768.0 for x in range(params.nframes) ])

        return pcmf32, pcmf32s

    def _output_txt(self, pcmf32s: List[List[float]]) -> Tuple[str, float]:
        n_segments = whisper_full_n_segments(self.context)
        result = ""
        prob = 0.0
        prob_n = 0
        for seg in range(n_segments):
            text = whisper_full_get_segment_text(self.context, seg)
            if self._params.language in {'zh', 'chinese'}:
                text = zhconv.convert(text, 'zh-cn')

            speaker = ""
            if self._params.diarize and len(pcmf32s) == 2:
                t0 = whisper_full_get_segment_t0(self.context, seg)
                t1 = whisper_full_get_segment_t1(self.context, seg)
                speaker = estimate_diarization_speaker(pcmf32s, t0, t1)

            result += speaker + text

            n_tokens = whisper_full_n_tokens(self.context, seg)
            for j in range(n_tokens):
                token = whisper_full_get_token_data(self.context, seg, j)
                prob += token.p
                prob_n += 1
        
        if prob_n > 0:
            prob /= prob_n

        # remove text pattern like (.*?) or [.*?] represent blank audio
        result = re.sub(self._blank_pattern1, '', result)
        result = re.sub(self._blank_pattern2, '', result)

        return result.strip(), prob

    def init_commands(self):
        cmd_chat = PROMPT_TEMPLATES['cmd_chat']

        self.k_prompt = cmd_chat['k_prompt']
        self.allowed_commands = cmd_chat['allowed_commands']

        cmd_func_list = [
            self.on_start_conversation,
            self.on_quit_command_mode,
        ]
        cmd_func_map = {}
        for x in cmd_func_list:
            cmd_func_map[x.__name__] = x
        
        logger.info(f'k_prompt: {self.k_prompt}')
        logger.info("allowed commands:")
        for cmd in self.allowed_commands:
            cmd['txt'] = cmd['txt'].lower()
            logger.info(f" - {cmd['txt']}")

            cmd['func'] = cmd_func_map.get(cmd['func'], None)

        self._cmd_stats = CmdStatus.NO_PROMPT
        self._pcmf32_prompt = []

    def process_commands(self, pcmf32: List[float]):
        start_tm = time.perf_counter()

        logprob_min = 0.0
        logprob_sum = 0.0
        n_tokens = 0

        wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        
        wparams.print_progress   = False
        wparams.print_special    = self._params.print_special
        wparams.print_realtime   = False
        wparams.print_timestamps = not self._params.no_timestamps
        wparams.translate        = True
        wparams.no_context       = True
        wparams.no_timestamps    = self._params.no_timestamps
        wparams.single_segment   = True
        wparams.max_tokens       = self._params.max_tokens
        wparams.language         = c_char_p('en'.encode('utf-8'))
        wparams.n_threads        = self._params.n_threads

        wparams.audio_ctx        = self._params.audio_ctx
        wparams.speed_up         = self._params.speed_up

        wparams.temperature     = 0.4
        wparams.temperature_inc = 1.0
        wparams.greedy.best_of  = 5

        wparams.beam_search.beam_size = 5

        wparams.initial_prompt = c_char_p(self._params.context.encode('utf-8'))

        code = whisper_full(self.context, wparams, pcmf32)
        if code != 0:
            raise WhisperException(f'whisper_full failed to process audio')
        
        n_segments = whisper_full_n_segments(self.context)
        result = ""
        for seg in range(n_segments):
            text = whisper_full_get_segment_text(self.context, seg)
            if self._params.language in {'zh', 'chinese'}:
                text = zhconv.convert(text, 'zh-cn')

            result += text

            n_tokens = whisper_full_n_tokens(self.context, seg)
            for j in range(n_tokens):
                token = whisper_full_get_token_data(self.context, seg, j)
                if token.plog > 0:
                    logger.error(f'token plog greater than zero: {token.plog}')
                    return

                logprob_min = min(logprob_min, token.plog)
                logprob_sum += token.plog
                n_tokens += 1

        result = re.sub(self._blank_pattern1, '', result)
        result = re.sub(self._blank_pattern2, '', result)
        result = result.strip().lower()

        elapsed = time.perf_counter() - start_tm
        if self._cmd_stats == CmdStatus.NO_PROMPT or self._cmd_stats == CmdStatus.CONVERSATION:
            prob = 100.0 * math.exp(logprob_min)
            sim = similarity(result, self.k_prompt)
            result_len = len(result)
            k_prompt_len = len(self.k_prompt)
            logger.info(f'Heard: [{result}] | prob: {prob:.06} | sim: {sim:.06} | elapsed: {elapsed:.03}s')

            if (result_len < (1 - WHISPER_PROMPT_LEN_DELTA) * k_prompt_len or 
                result_len > (1 + WHISPER_PROMPT_LEN_DELTA) * k_prompt_len or 
                sim < WHISPER_PROMPT_SIM_THOLD
            ):
                if self._cmd_stats == CmdStatus.NO_PROMPT:
                    logger.info('prompt not recognized, try again.')
                    play_const_audio('PROMPT_NOT_RECOGNIZED')

                return
            
            logger.info('The prompt has been recognized! Waiting for voice commands ...')
            
            self.on_enter_command_mode()
            self._pcmf32_prompt = pcmf32
            # self._pcmf32_prompt = pcmf32 + [0.0] * (AudioRecorder.RATE * 3)

        else:
            prob = 100.0 * math.exp(logprob_min)

            best_sim = 0.0
            best_len = 0
            result_len = len(result)
            k_prompt_len = len(self.k_prompt)
            for n in range(int(0.8 * k_prompt_len), int(1.2 * k_prompt_len) + 1):
                if n >= result_len:
                    break
                prompt = result[0:n]
                sim = similarity(prompt, self.k_prompt)
                if sim > best_sim:
                    best_sim = sim
                    best_len = n

            logger.info(f'result: [{result}], prob: {prob:.02}% | best_sim: {best_sim:.06}, best_len: {best_len} | elapsed: {elapsed:.03}s')
            
            if best_len == 0:
                logger.info('prompt not recognized, try again.')
                return
            
            command = result[best_len:].strip()
            logger.info(f'command: [{command}]')

            max_sim = float('-inf')
            best_cmd = None
            for cmd in self.allowed_commands:
                sim = similarity(command, cmd['txt'])
                logger.info(f"- {cmd['txt']}: sim({sim:.06})")

                if sim > max_sim:
                    max_sim = sim
                    best_cmd = cmd
            logger.info(f'best_cmd: {best_cmd}, max_sim: {max_sim:.06}')
            if max_sim > WHISPER_COMMAND_SIM_THOLD:
                func = best_cmd['func']
                if func:
                    func()
            else:
                play_const_audio('NO_CMD_MATCHED')

    def on_enter_command_mode(self):
        logger.info('on_enter_command_mode')
        self._cmd_stats = CmdStatus.HAVE_PROMPT
        play_const_audio('PROMPT_RECOGNIZED')

    def on_start_conversation(self):
        logger.info('on_start_conversation')
        self._cmd_stats = CmdStatus.CONVERSATION
        play_const_audio('START_CONVERSATION')
    
    def on_quit_command_mode(self):
        logger.info('on_quit_command_mode')
        self._cmd_stats = CmdStatus.NO_PROMPT
        play_const_audio('ON_BYE')

    def __del__(self):
        whisper_print_timings(self.context)
        # free the memory
        print('whisper_free context')
        whisper_free(self.context)
