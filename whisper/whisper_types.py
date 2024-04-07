from enum import Enum, unique

class WhisperParams():
    n_threads    = 4
    n_processors =  1
    offset_t_ms  =  0
    offset_n     =  0
    duration_ms  =  0
    progress_step = 5
    max_context  = -1
    max_len      =  0
    best_of      = -1
    beam_size    = -1
    audio_ctx   = 0

    word_thold    =  0.01
    entropy_thold =  2.40
    logprob_thold = -1.00

    speed_up        = False
    debug_mode      = False
    translate       = False
    detect_language = False
    diarize         = False
    tinydiarize     = False
    split_on_word   = False
    no_fallback     = False
    output_txt      = True
    output_vtt      = False
    output_srt      = False
    output_wts      = False
    output_csv      = False
    output_jsn      = False
    output_jsn_full = False
    output_lrc      = False
    no_prints       = False
    print_special   = False
    print_colors    = False
    print_progress  = False
    no_timestamps   = False
    log_score       = False
    use_gpu         = True
    max_tokens      = 32
    context         = ''

    language  = "en"
    prompt = ''
    font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"
    model     = "models/ggml-base.en.bin"

    # [TDRZ] speaker turn string
    tdrz_speaker_turn = " [SPEAKER_TURN]"

    openvino_encode_device = "CPU"

class WhisperException(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return f'WhisperException: {self.message}'
