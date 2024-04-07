from typing import List, Tuple, Dict, Literal, Optional
from pydantic import BaseModel, Field
from configs import (
    EMBEDDING_MODEL, 
    CHUNK_SIZE, 
    OVERLAP_SIZE, 
    ZH_TITLE_ENHANCE, 
    DEFAULT_VS_TYPE,
    VECTOR_SEARCH_TOP_K,
    SCORE_THRESHOLD,
)

class WsBase(BaseModel):
    action: str

class WsResponse(WsBase):
    code: int = 0
    msg: Optional[str] = 'success'
    data: Optional[object] = None

class KBListData(BaseModel):
    data: List[str]

class KBCreateRequest(WsBase):
    kb_name: str # knowledge base name
    vs_type: Optional[str] = 'faiss'
    model: Optional[str] = EMBEDDING_MODEL

class KBDeleteRequest(WsBase):
    kb_name: str # knowledge base name

class KBUpdateInfoRequest(WsBase):
    kb_name: str # knowledge base name
    kb_info: str

class KBListFilesRequest(WsBase):
    kb_name: str # knowledge base name

class KBListFilesData(BaseModel):
    data: List[str]

class KBUpdateDocsRequest(WsBase):
    kb_name: str # knowledge base name
    file_names: List[str] # file list
    chunk_size: Optional[int] = CHUNK_SIZE # Max length for a chunk
    chunk_overlap: Optional[int] = OVERLAP_SIZE # overlap for tow chunks nearby
    zh_title_enhance: Optional[bool] = ZH_TITLE_ENHANCE # enhance chinese title processing
    not_refresh_vs_cache: Optional[bool] = False # not refresh vector store

class KBUpdateDocsData(BaseModel):
    failed_files: Dict

class KBDeleteDocsRequest(WsBase):
    kb_name: str # knowledge base name
    file_names: List[str] # file list
    delete_content: Optional[bool] = False # delete files in disk or not
    not_refresh_vs_cache: Optional[bool] = False # not refresh vector store

class KBDeleteDocsData(BaseModel):
    failed_files: Dict

class KBRecreateVectorStoreRequest(WsBase):
    kb_name: str # knowledge base name
    vs_type: Optional[str] = DEFAULT_VS_TYPE
    embed_model: Optional[str] = EMBEDDING_MODEL
    chunk_size: Optional[int] = CHUNK_SIZE # Max length for a chunk
    chunk_overlap: Optional[int] = OVERLAP_SIZE # overlap for tow chunks nearby
    zh_title_enhance: Optional[bool] = ZH_TITLE_ENHANCE # enhance chinese title processing
    not_refresh_vs_cache: Optional[bool] = False # not refresh vector store

class KBRecreateVectorStoreData(BaseModel):
    total: int # 文件总数
    finished: int # 已完成文件数
    doc: str # 当前完成的文件名

class KBSearchDocsRequest(WsBase):
    query: str # user input
    kb_name: str # knowledge base name
    top_k: Optional[int] = VECTOR_SEARCH_TOP_K # vector matched
    score_threshold: Optional[float] = SCORE_THRESHOLD # from 0 to 1, 0.5 is recommended
    file_name: Optional[str] = ""
    metadata: Optional[dict] = {}

class SearchDocsFromKBRequest(WsBase):
    kb_type: int
    query: str # user input

###########################################################

class BaseResponse(BaseModel):
    code: int = 0
    msg: Optional[str] = 'success'
    data: Optional[object] = None

class KBDownloadDocRequest(BaseModel):
    kb_name: str
    file_name: str
    preview: Optional[bool] = False

###########################################################

class ApiSearchDocsRequest(BaseModel):
    kb_name: str
    query: str

class ApiSearchDocsResponse(BaseModel):
    chunks: List

