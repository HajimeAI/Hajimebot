from typing import List, Tuple, Dict, Literal, Optional, Union
from pydantic import BaseModel, Field

class CentralWsBase(BaseModel):
    msgType: Optional[str] = None

class CentralWsRspDataBase(BaseModel):
    code: int = 0
    msg: Optional[str] = 'success'

class LoginInfoData(BaseModel):
    id: int
    type: int
    cmd: str
    ip: str

class LoginInfo(CentralWsBase):
    data: LoginInfoData

class QueryIndexedData(BaseModel):
    id: int
    name: str

class QueryIndexed(CentralWsBase):
    data: List[QueryIndexedData]

class HeartBeatData(BaseModel):
    imei: str

class HeartBeat(CentralWsBase):
    data: HeartBeatData

class FileTransferData(BaseModel):
    fileName: str

class FileTransfer(CentralWsBase):
    data: FileTransferData

class FileTransferResponse(CentralWsBase):
    data: CentralWsRspDataBase

class FileDelData(BaseModel):
    fileName: str

class FileDel(CentralWsBase):
    data: FileDelData

class FileDelResponse(CentralWsBase):
    data: CentralWsRspDataBase

class QueryData(BaseModel):
    imei: str
    type: int

class QueryRequest(CentralWsBase):
    data: QueryData

class QueryResponseData(BaseModel):
    imei: str
    ip: str
    type: int

class QueryResponse(CentralWsBase):
    data: List[QueryResponseData]

class LogData(BaseModel):
    imei: str
    content: str

class LogRequest(CentralWsBase):
    data: LogData

class NodeDetailData(BaseModel):
    imei: str
    hw_info: Dict
    compute_cap_level: int
    compute_cap: Dict

class NodeDetailRequest(CentralWsBase):
    data: NodeDetailData

class EventData(BaseModel):
    kind: str
    node_id: str
    content: Union[Dict, str]

class EventRequest(CentralWsBase):
    data: EventData
