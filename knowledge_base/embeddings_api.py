from langchain.docstore.document import Document
from typing import Dict, List

from configs import EMBEDDING_MODEL, logger
from utils import BaseResponse
from knowledge_base.kb_cache.base import load_local_embeddings


def embed_texts(
    texts: List[str],
    embed_model: str = EMBEDDING_MODEL,
) -> BaseResponse:
    try:
        embeddings = load_local_embeddings(model=embed_model)
        return BaseResponse(data=embeddings.embed_documents(texts))
            

    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"vectorize error: {e}")


async def aembed_texts(
    texts: List[str],
    embed_model: str = EMBEDDING_MODEL,
) -> BaseResponse:
    try:
        embeddings = load_local_embeddings(model=embed_model)
        return BaseResponse(data=await embeddings.aembed_documents(texts))

    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"vectorize error: {e}")

def embed_documents(
    docs: List[Document],
    embed_model: str = EMBEDDING_MODEL,
) -> Dict:
    texts = [x.page_content for x in docs]
    metadatas = [x.metadata for x in docs]
    embeddings = embed_texts(texts=texts, embed_model=embed_model).data
    if embeddings is not None:
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
