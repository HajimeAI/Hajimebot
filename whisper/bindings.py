import os
from ctypes import *
from typing import List, Tuple

cur_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    whisper_cpp = CDLL(f'{cur_dir}/whisper.dll', winmode=0)
else:
    whisper_cpp = CDLL(f'{cur_dir}/libwhisper.so')

WHISPER_SAMPLING_GREEDY = 0
WHISPER_SAMPLING_BEAM_SEARCH = 1

class whisper_context_params(Structure):
    _fields_ = [
        ('use_gpu', c_bool),
        ('gpu_device', c_int),
    ]

class cls_greedy(Structure):
    _fields_ = [
        ('best_of', c_int),
    ]

class cls_beam_search(Structure):
    _fields_ = [
        ('beam_size', c_int),
        ('patience', c_float),
    ]

class whisper_print_user_data(Structure):
    _fields_ = [
        ('params', py_object),
        ('pcmf32s', py_object),
        ('progress_prev', c_int),
    ]

WHISPER_NEW_SEQ_CALLBACK = CFUNCTYPE(None, c_void_p, c_void_p, c_int, POINTER(whisper_print_user_data))

GGML_LOG_CALLBACK = CFUNCTYPE(None, c_int, c_void_p, c_void_p)

class whisper_full_params(Structure):
    _fields_ = [
        ('strategy', c_int),
        ('n_threads', c_int),
        ('n_max_text_ctx', c_int),    # max tokens to use from past text as prompt for the decoder
        ('offset_ms', c_int),         # start offset in ms
        ('duration_ms', c_int),       # audio duration to process in ms

        ('translate', c_bool),
        ('no_context', c_bool),       # do not use past transcription (if any) as initial prompt for the decoder
        ('no_timestamps', c_bool),    # do not generate timestamps
        ('single_segment', c_bool),   # force single segment output (useful for streaming)
        ('print_special', c_bool),    # print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
        ('print_progress', c_bool),   # print progress information
        ('print_realtime', c_bool),   # print results from within whisper.cpp (avoid it, use callback instead)
        ('print_timestamps', c_bool), # print timestamps for each text segment when printing realtime

        # [EXPERIMENTAL] token-level timestamps
        ('token_timestamps', c_bool),    # enable token-level timestamps
        ('thold_pt', c_float),           # timestamp token probability threshold (~0.01)
        ('thold_ptsum', c_float),        # timestamp token sum probability threshold (~0.01)
        ('max_len', c_int),              # max segment length in characters
        ('split_on_word', c_bool),       # split on word rather than on token (when used with max_len)
        ('max_tokens', c_int),           # max tokens per segment (0 = no limit)

        # [EXPERIMENTAL] speed-up techniques
        # note: these can significantly reduce the quality of the output
        ('speed_up', c_bool),       # speed-up the audio by 2x using Phase Vocoder
        ('audio_ctx', c_int),       # overwrite the audio context size (0 = use default)

        # [EXPERIMENTAL] [TDRZ] tinydiarize
        ('tdrz_enable', c_bool),       # enable tinydiarize speaker turn detection

        # tokens to provide to the whisper decoder as initial prompt
        # these are prepended to any existing text context from a previous call
        ('initial_prompt', c_char_p),
        ('prompt_tokens', c_void_p),
        ('prompt_n_tokens', c_int),

        # for auto-detection, set to nullptr, "" or "auto"
        ('language', c_char_p),
        ('detect_language', c_bool), 

        # common decoding parameters:
        ('suppress_blank', c_bool),              # ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
        ('suppress_non_speech_tokens', c_bool),  # ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253

        ('temperature', c_float),        # initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
        ('max_initial_ts', c_float),     # ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
        ('length_penalty', c_float),     # ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267

        # fallback parameters
        # ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278
        ('temperature_inc', c_float),
        ('entropy_thold', c_float),     # similar to OpenAI's "compression_ratio_threshold"
        ('logprob_thold', c_float),
        ('no_speech_thold', c_float),   # TODO: not implemented

        ('greedy', cls_greedy), # ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L264

        # ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L265
        # TODO: not implemented, ref: https://arxiv.org/pdf/2204.05424.pdf
        ('beam_search', cls_beam_search),

        # called for every newly generated text segment
        ('new_segment_callback', WHISPER_NEW_SEQ_CALLBACK),
        ('new_segment_callback_user_data', POINTER(whisper_print_user_data)),

        # called on each progress update
        ('progress_callback', c_void_p),
        ('progress_callback_user_data', c_void_p),

        # called each time before the encoder starts
        ('encoder_begin_callback', c_void_p),
        ('encoder_begin_callback_user_data', c_void_p),

        # called each time before ggml computation starts
        ('abort_callback', c_void_p),
        ('abort_callback_user_data', c_void_p),

        # called by each decoder to filter obtained logits
        ('logits_filter_callback', c_void_p),
        ('logits_filter_callback_user_data', c_void_p),

        ('grammar_rules', c_void_p),
        ('n_grammar_rules', c_size_t),
        ('i_start_rule', c_size_t),
        ('grammar_penalty', c_float),
    ]

class whisper_token_data(Structure):
    _fields_ = [
        ('id', c_int),  # token id
        ('tid', c_int), # forced timestamp token id

        ('p', c_float),      # probability of the token
        ('plog', c_float),   # log probability of the token
        ('pt', c_float),     # probability of the timestamp token
        ('ptsum', c_float),  # sum of probabilities of all timestamp tokens

        # token-level timestamp data
        # do not use if you haven't computed token-level timestamps
        ('t0', c_int64),        # start time of the token
        ('t1', c_int64),        # end time of the token

        ('vlen', c_float),  # voice length of the token
    ]

whisper_cpp.whisper_lang_id.argtypes = [ c_char_p ]
whisper_cpp.whisper_lang_id.restype = c_int
def whisper_lang_id(lang: str) -> int:
    return whisper_cpp.whisper_lang_id(c_char_p(lang.encode('utf-8')))

whisper_cpp.whisper_init_from_file_with_params.argtypes = [ c_char_p ]
whisper_cpp.whisper_init_from_file_with_params.restype = c_void_p
def whisper_init_from_file_with_params(model_path: str, params: whisper_context_params) -> int:
    return whisper_cpp.whisper_init_from_file_with_params(c_char_p(model_path.encode('utf-8')), params)

# whisper_cpp.whisper_print_system_info.argtypes = [ ]
whisper_cpp.whisper_print_system_info.restype = c_char_p
def whisper_print_system_info() -> str:
    _text = whisper_cpp.whisper_print_system_info()
    return _text.decode('utf-8')

whisper_cpp.whisper_is_multilingual.argtypes = [ c_void_p ]
whisper_cpp.whisper_is_multilingual.restype = c_int
def whisper_is_multilingual(ctx: int) -> int:
    return whisper_cpp.whisper_is_multilingual(ctx)

whisper_cpp.whisper_context_default_params.restype = whisper_context_params
def whisper_context_default_params() -> whisper_context_params:
    return whisper_cpp.whisper_context_default_params()

whisper_cpp.whisper_full_default_params.argtypes = [ c_int ]
whisper_cpp.whisper_full_default_params.restype = whisper_full_params
def whisper_full_default_params(strategy: int) -> whisper_full_params:
    return whisper_cpp.whisper_full_default_params(strategy)

whisper_cpp.whisper_full_parallel.argtypes = [ c_void_p, whisper_full_params, c_void_p, c_int, c_int ]
whisper_cpp.whisper_full_parallel.restype = c_int
def whisper_full_parallel(ctx: int, params: whisper_full_params, samples: List[float], n_processors: int) -> int:
    n_samples = len(samples)
    samples_array = c_float * n_samples
    _samples = samples_array()
    for i in range(n_samples):
        _samples[i] = samples[i]
    return whisper_cpp.whisper_full_parallel(c_void_p(ctx), params, _samples, n_samples, n_processors)

whisper_cpp.whisper_full.argtypes = [ c_void_p, whisper_full_params, c_void_p, c_int ]
whisper_cpp.whisper_full.restype = c_int
def whisper_full(ctx: int, params: whisper_full_params, samples: List[float]) -> int:
    n_samples = len(samples)
    samples_array = c_float * n_samples
    _samples = samples_array()
    for i in range(n_samples):
        _samples[i] = samples[i]
    return whisper_cpp.whisper_full(c_void_p(ctx), params, _samples, n_samples)

whisper_cpp.whisper_full_n_segments.argtypes = [ c_void_p ]
whisper_cpp.whisper_full_n_segments.restype = c_int
def whisper_full_n_segments(ctx: int) -> int:
    return whisper_cpp.whisper_full_n_segments(c_void_p(ctx))

whisper_cpp.whisper_full_get_segment_t0.argtypes = [ c_void_p, c_int ]
whisper_cpp.whisper_full_get_segment_t0.restype = c_int64
def whisper_full_get_segment_t0(ctx: int, i_segment: int) -> int:
    return whisper_cpp.whisper_full_get_segment_t0(c_void_p(ctx), i_segment)

whisper_cpp.whisper_full_get_segment_t1.argtypes = [ c_void_p, c_int ]
whisper_cpp.whisper_full_get_segment_t1.restype = c_int64
def whisper_full_get_segment_t1(ctx: int, i_segment: int) -> int:
    return whisper_cpp.whisper_full_get_segment_t1(c_void_p(ctx), i_segment)

whisper_cpp.whisper_full_get_segment_text.argtypes = [ c_void_p, c_int ]
whisper_cpp.whisper_full_get_segment_text.restype = c_char_p
def whisper_full_get_segment_text(ctx: int, i_segment: int) -> str:
    _text = whisper_cpp.whisper_full_get_segment_text(c_void_p(ctx), i_segment)
    try:
        return _text.decode('utf-8')
    except UnicodeDecodeError:
        return _text.decode('iso-8859-1')

whisper_cpp.whisper_full_n_tokens.argtypes = [ c_void_p, c_int ]
whisper_cpp.whisper_full_n_tokens.restype = c_int
def whisper_full_n_tokens(ctx: int, i_segment: int) -> int:
    return whisper_cpp.whisper_full_n_tokens(c_void_p(ctx), i_segment)

whisper_cpp.whisper_full_get_token_data.argtypes = [ c_void_p, c_int, c_int ]
whisper_cpp.whisper_full_get_token_data.restype = whisper_token_data
def whisper_full_get_token_data(ctx: int, i_segment: int, i_token: int) -> whisper_token_data:
    return whisper_cpp.whisper_full_get_token_data(c_void_p(ctx), i_segment, i_token)

whisper_cpp.whisper_full_get_segment_speaker_turn_next.argtypes = [ c_void_p, c_int ]
whisper_cpp.whisper_full_get_segment_speaker_turn_next.restype = c_bool
def whisper_full_get_segment_speaker_turn_next(ctx: int, i_segment: int) -> bool:
    return whisper_cpp.whisper_full_get_segment_speaker_turn_next(c_void_p(ctx), i_segment)

whisper_cpp.whisper_full_get_token_id.argtypes = [ c_void_p, c_int, c_int ]
whisper_cpp.whisper_full_get_token_id.restype = c_int
def whisper_full_get_token_id(ctx: int, i_segment: int, i_token: int) -> int:
    return whisper_cpp.whisper_full_get_token_id(c_void_p(ctx), i_segment, i_token)

whisper_cpp.whisper_full_get_token_text.argtypes = [ c_void_p, c_int, c_int ]
whisper_cpp.whisper_full_get_token_text.restype = c_char_p
def whisper_full_get_token_text(ctx: int, i_segment: int, i_token: int) -> str:
    _text: bytes = whisper_cpp.whisper_full_get_token_text(c_void_p(ctx), i_segment, i_token)
    try:
        return _text.decode('utf-8')
    except UnicodeDecodeError:
        return _text.decode('iso-8859-1')

whisper_cpp.whisper_full_get_token_p.argtypes = [ c_void_p, c_int, c_int ]
whisper_cpp.whisper_full_get_token_p.restype = c_float
def whisper_full_get_token_p(ctx: int, i_segment: int, i_token: int) -> float:
    return whisper_cpp.whisper_full_get_token_p(c_void_p(ctx), i_segment, i_token)

whisper_cpp.whisper_token_eot.argtypes = [ c_void_p ]
whisper_cpp.whisper_token_eot.restype = c_int
def whisper_token_eot(ctx: int) -> int:
    return whisper_cpp.whisper_token_eot(c_void_p(ctx))

whisper_cpp.whisper_model_type_readable.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_type_readable.restype = c_char_p
def whisper_model_type_readable(ctx: int) -> str:
    _text = whisper_cpp.whisper_model_type_readable(c_void_p(ctx))
    return _text.decode('utf-8')

whisper_cpp.whisper_model_n_vocab.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_vocab.restype = c_int
def whisper_model_n_vocab(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_vocab(c_void_p(ctx))

whisper_cpp.whisper_model_n_audio_ctx.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_audio_ctx.restype = c_int
def whisper_model_n_audio_ctx(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_audio_ctx(c_void_p(ctx))

whisper_cpp.whisper_model_n_audio_state.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_audio_state.restype = c_int
def whisper_model_n_audio_state(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_audio_state(c_void_p(ctx))

whisper_cpp.whisper_model_n_audio_head.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_audio_head.restype = c_int
def whisper_model_n_audio_head(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_audio_head(c_void_p(ctx))

whisper_cpp.whisper_model_n_audio_layer.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_audio_layer.restype = c_int
def whisper_model_n_audio_layer(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_audio_layer(c_void_p(ctx))

whisper_cpp.whisper_model_n_text_ctx.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_text_ctx.restype = c_int
def whisper_model_n_text_ctx(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_text_ctx(c_void_p(ctx))

whisper_cpp.whisper_model_n_text_state.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_text_state.restype = c_int
def whisper_model_n_text_state(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_text_state(c_void_p(ctx))

whisper_cpp.whisper_model_n_text_head.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_text_head.restype = c_int
def whisper_model_n_text_head(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_text_head(c_void_p(ctx))

whisper_cpp.whisper_model_n_text_layer.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_text_layer.restype = c_int
def whisper_model_n_text_layer(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_text_layer(c_void_p(ctx))

whisper_cpp.whisper_model_n_mels.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_n_mels.restype = c_int
def whisper_model_n_mels(ctx: int) -> int:
    return whisper_cpp.whisper_model_n_mels(c_void_p(ctx))

whisper_cpp.whisper_model_ftype.argtypes = [ c_void_p ]
whisper_cpp.whisper_model_ftype.restype = c_int
def whisper_model_ftype(ctx: int) -> int:
    return whisper_cpp.whisper_model_ftype(c_void_p(ctx))

whisper_cpp.whisper_full_lang_id.argtypes = [ c_void_p ]
whisper_cpp.whisper_full_lang_id.restype = c_int
def whisper_full_lang_id(ctx: int) -> int:
    return whisper_cpp.whisper_full_lang_id(c_void_p(ctx))

whisper_cpp.whisper_lang_str.argtypes = [ c_int ]
whisper_cpp.whisper_lang_str.restype = c_char_p
def whisper_lang_str(lang_id: int) -> str:
    _text = whisper_cpp.whisper_lang_str(lang_id)
    return _text.decode('utf-8')

whisper_cpp.whisper_tokenize.argtypes = [ c_void_p, c_char_p, c_void_p, c_int ]
whisper_cpp.whisper_tokenize.restype = c_int
def whisper_tokenize(ctx: int, text: str, max_tokens: int) -> Tuple[int, Array]:
    tokens_array = c_int * max_tokens
    tokens = tokens_array()
    num_tokens_success = whisper_cpp.whisper_tokenize(
        c_void_p(ctx),
        c_char_p(text.encode('utf-8')),
        tokens,
        max_tokens
    )
    return num_tokens_success, tokens

whisper_cpp.whisper_get_logits.argtypes = [ c_void_p ]
whisper_cpp.whisper_get_logits.restype = POINTER(c_float)
def whisper_get_logits(ctx: int) -> POINTER:
    return whisper_cpp.whisper_get_logits(c_void_p(ctx))

whisper_cpp.whisper_n_vocab.argtypes = [ c_void_p ]
whisper_cpp.whisper_n_vocab.restype = c_int
def whisper_n_vocab(ctx: int) -> int:
    return whisper_cpp.whisper_n_vocab(c_void_p(ctx))

whisper_cpp.whisper_token_to_str.argtypes = [ c_void_p, c_int ]
whisper_cpp.whisper_token_to_str.restype = c_char_p
def whisper_token_to_str(ctx: int, token: int) -> str:
    _text = whisper_cpp.whisper_token_to_str(c_void_p(ctx), token)
    return _text.decode('utf-8')

whisper_cpp.whisper_print_timings.argtypes = [ c_void_p ]
def whisper_print_timings(ctx: int):
    whisper_cpp.whisper_print_timings(c_void_p(ctx))

whisper_cpp.whisper_log_set.argtypes = [ c_void_p, c_void_p ]
def whisper_log_set(log_callback: c_void_p, user_data: c_void_p):
    whisper_cpp.whisper_log_set(log_callback, user_data)

whisper_cpp.whisper_free.argtypes = [ c_void_p ]
def whisper_free(ctx: int):
    whisper_cpp.whisper_free(c_void_p(ctx))
