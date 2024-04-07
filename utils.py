import pydantic
from pydantic import BaseModel
from typing import List
from pathlib import Path
from configs import (
    EMBEDDING_DEVICE,
    MODEL_ROOT_PATH, 
    logger, 
    log_verbose,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (
    Literal,
    Optional,
    Callable,
    Generator,
    Dict,
    Any,
)

def detect_device() -> Literal["cuda", "mps", "cpu"]:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"

def embedding_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    device = device or EMBEDDING_DEVICE
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device

def get_model_path(model_name: str, type: str = None) -> Optional[str]:
    path_str = model_name
    path = Path(path_str)
    if path.is_dir():
        return str(path)

    root_path = Path(MODEL_ROOT_PATH)
    if root_path.is_dir():
        path = root_path / model_name
        if path.is_dir():  # use key, {MODEL_ROOT_PATH}/chatglm-6b
            return str(path)
        path = root_path / path_str
        if path.is_dir():  # use value, {MODEL_ROOT_PATH}/THUDM/chatglm-6b-new
            return str(path)
        path = root_path / path_str.split("/")[-1]
        if path.is_dir():  # use value split by "/", {MODEL_ROOT_PATH}/chatglm-6b-new
            return str(path)
    return path_str  # THUDM/chatglm06b

def torch_gc():
    try:
        import torch
        if torch.cuda.is_available():
            # with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
            try:
                from torch.mps import empty_cache
                empty_cache()
            except Exception as e:
                msg = "Up to pytorch 2.0.0 in macOS is required."
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
    except Exception:
        ...

def run_in_thread_pool(
    func: Callable,
    params: List[Dict] = [],
) -> Generator:
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            thread = pool.submit(func, **kwargs)
            tasks.append(thread)

        for obj in as_completed(tasks):
            yield obj.result()


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }
