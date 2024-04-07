import os
from enum import Enum
from .kb import MODEL_ROOT_PATH, KB_ROOT_PATH

NODE_COMPUTE_CAP_LEVEL = 2
NODE_COMPUTE_CAP_INFO = {
    'level_1': {
        'stt': True,
        'tts': False,
        'llm': False,
        'ai_modules': ['Whisper', ],
    },
    'level_2': {
        'stt': True,
        'tts': False,
        'llm': False,
        'ai_modules': ['Whisper', 'FAISS', ],
    },
    'level_3': {
        'stt': True,
        'tts': True,
        'llm': True,
        'ai_modules': ['Whisper', 'FAISS', 'OLLAMA', 'GPT-soVITS'],
    },
}

# Whisper model
WHISPER_MODEL = 'ggml-medium.bin'
WHISPER_MODEL_DIR = os.path.join(MODEL_ROOT_PATH, "whisper")
WHISPER_MODEL_PATH = os.path.join(WHISPER_MODEL_DIR, WHISPER_MODEL)

# Node info
NODE_INFO_PATH = os.path.join(KB_ROOT_PATH, 'node_info.pickle')

# Central server URL
CENTRAL_SERVER_HOST = '18.181.169.40'
CENTRAL_SERVER_WS_URL = f'ws://{CENTRAL_SERVER_HOST}/ws/instance'
CENTRAL_SERVER_DOWNLOAD_URL = f'http://{CENTRAL_SERVER_HOST}/download'
# CENTRAL_SERVER_PROXIES = {
#     "http": "http://127.0.0.1:8118",
#     "https": "https://127.0.0.1:8118",
# }
CENTRAL_SERVER_PROXIES = None
# OPENAI_PROXY = 'http://127.0.0.1:8118'
OPENAI_PROXY = None
WS_MAX_SIZE = 10 * 1024 * 1024

# tts
SPEECH_DIR = os.path.join(KB_ROOT_PATH, "speech")

# local tts
USING_LOCAL_TTS = False
LOCAL_TTS_SERVER = 'http://127.0.0.1:5000/tts'
DEFAULT_VOICE_LOCAL = '阮•梅'

# const speech
SPEECH_DELAY_SECONDS = 5
DEFAULT_VOICE = 'en-US-EmmaMultilingualNeural'
# DEFAULT_VOICE = 'zh-CN-XiaoxiaoNeural'
AUDIO_FILE_SUFFIX = 'wav' if USING_LOCAL_TTS else 'mp3'

CONST_SPEECH_MAP = {
    'BOOTED': {
        'txt': 'HajimeBot start suceessfully.',
        'file': os.path.join(SPEECH_DIR, f"booted_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'CONNECTED': {
        'txt': 'Successfully accessed the AI computing power network.',
        'file': os.path.join(SPEECH_DIR, f"connected_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'PROMPT_RECOGNIZED': {
        'txt': 'May I help you?',
        'file': os.path.join(SPEECH_DIR, f"prompt_recognized_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'PROMPT_NOT_RECOGNIZED': {
        'txt': 'Please provide correct prompt.',
        'file': os.path.join(SPEECH_DIR, f"prompt_not_recognized_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'START_CONVERSATION': {
        'txt': 'Glad to hear your voice.',
        'file': os.path.join(SPEECH_DIR, f"start_conversation_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'ON_BYE': {
        'txt': 'Good bye. Have a nice day.',
        'file': os.path.join(SPEECH_DIR, f"on_bye_speech.{AUDIO_FILE_SUFFIX}"),
    },
    'NO_CMD_MATCHED': {
        'txt': 'I can\'t recognized your command.',
        'file': os.path.join(SPEECH_DIR, f"no_cmd_matched_speech.{AUDIO_FILE_SUFFIX}"),
    },
}

# Knowledge base searching selection
INDEX_TYPE_SELECTION = 'solana'

# local llm
LOCAL_LLM_SERVER = 'http://10.10.10.12:11434'
LOCAL_LLM_MODEL = 'gemma:7b' # llama2:13b

class ChatModels(Enum):
    OPENAI = 0,
    OLLAMA = 1,

DEFAULT_CHAT_MODEL = ChatModels.OLLAMA

WHISPER_PROMPT_LEN_DELTA = 0.4
WHISPER_PROMPT_SIM_THOLD = 0.5
WHISPER_COMMAND_SIM_THOLD = 0.6
