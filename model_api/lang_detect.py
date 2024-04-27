import os
from configs import MODEL_ROOT_PATH
os.environ['FTLANG_CACHE'] = os.path.join(MODEL_ROOT_PATH, "fasttext-langdetect")

from ftlangdetect.detect import get_or_load_model, detect
from configs import DEFAULT_VOICE, DEFAULT_VOICE_LOCAL_MAP, USING_LOCAL_TTS

def load_model(low_memory=False):
    get_or_load_model(low_memory)

def text_detect(text: str, low_memory=False):
    return detect(text, low_memory)

def get_voice_by_lang(text: str):
    if USING_LOCAL_TTS:
        detect_result = text_detect(text)
        lang = detect_result['lang']
        if lang in DEFAULT_VOICE_LOCAL_MAP:
            return DEFAULT_VOICE_LOCAL_MAP[lang]
        else:
            return DEFAULT_VOICE_LOCAL_MAP['default']
    else:
        return DEFAULT_VOICE