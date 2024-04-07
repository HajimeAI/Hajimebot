import os

# knowledge base path
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
if not os.path.exists(KB_ROOT_PATH):
    os.mkdir(KB_ROOT_PATH)
# database path
DB_ROOT_PATH = os.path.join(KB_ROOT_PATH, "knowledge_base.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_ROOT_PATH}"

# model root path
MODEL_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# embedding model
EMBEDDING_MODEL = "bge-large-en-v1.5"
EMBEDDING_MODEL_PATH = os.path.join(MODEL_ROOT_PATH, EMBEDDING_MODEL)

# device to run embedding model, "cuda", "mps", "cpu"
EMBEDDING_DEVICE = "cuda"

# default knowledge base
DEFAULT_KNOWLEDGE_BASE = "samples"

# default vector store type
DEFAULT_VS_TYPE = "faiss"

# cached vector store number, for faiss
CACHED_VS_NUM = 1

# temp cached vector store number, for faiss
CACHED_MEMO_VS_NUM = 10

# vector store chunk size
CHUNK_SIZE = 250

# vector store overlap size
OVERLAP_SIZE = 50

# vector store searching number
VECTOR_SEARCH_TOP_K = 3

# vector store match threshold
SCORE_THRESHOLD = 1.0

# start chinese title enhancing or not
ZH_TITLE_ENHANCE = False

# PDF OCR control threshold
PDF_OCR_THRESHOLD = (0.6, 0.6)

# nltk path
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

# TextSplitter config
text_splitter_dict = {
    "ChineseRecursiveTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": EMBEDDING_MODEL_PATH,
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on": [
            ("#", "head1"),
            ("##", "head2"),
            ("###", "head3"),
            ("####", "head4"),
        ],
    },
}

# TEXT_SPLITTER name
TEXT_SPLITTER_NAME = "RecursiveCharacterTextSplitter"

# Embedding keywords
EMBEDDING_KEYWORD_FILE = "embedding_keywords.txt"
