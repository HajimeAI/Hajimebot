import urllib

from configs import logger, log_verbose
from communication.models import *
from communication.manager.local_manager import g_local_manager
from db.repository.knowledge_base_repository import list_kbs_from_db
from knowledge_base.utils import validate_kb_name
from knowledge_base.kb_service.base import KBServiceFactory

def list_kbs(req: WsBase) -> WsResponse:
    # Get List of Knowledge Base
    return WsResponse(
        action=req.action,
        data=KBListData(
            data=list_kbs_from_db(),
        )
    )

def create_kb(req: KBCreateRequest) -> WsResponse:
    # Create selected knowledge base
    if not validate_kb_name(req.kb_name):
        return WsResponse(action=req.action, code=403, msg="Don't attack me")
    if req.kb_name.strip() == "":
        return WsResponse(action=req.action, code=404, msg="Knowledge base name is empty")

    kb = KBServiceFactory.get_service_by_name(req.kb_name)
    if kb is not None:
        return WsResponse(action=req.action, code=-1)

    kb = KBServiceFactory.get_service(req.kb_name, req.vs_type, req.model)
    try:
        kb.create_kb()
    except Exception as e:
        msg = f"Knowledge base creation failed: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e if log_verbose else None)
        return WsResponse(action=req.action, code=500, msg=msg)

    return WsResponse(action=req.action)

def delete_kb(req: KBDeleteRequest) -> WsResponse:
    # Delete selected knowledge base
    if not validate_kb_name(req.kb_name):
        return WsResponse(action=req.action, code=403, msg="Don't attack me")
    req.kb_name = urllib.parse.unquote(req.kb_name)

    kb = KBServiceFactory.get_service_by_name(req.kb_name)
    if kb is None:
        return WsResponse(action=req.action, code=404, msg=f"Knowledge base '{req.kb_name}' not found.")

    try:
        status = kb.clear_vs()
        status = kb.drop_kb()
        if status:
            return WsResponse(
                action=req.action, 
            )
    except Exception as e:
        msg = f"Knowledge base deletion failed: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}', exc_info=e if log_verbose else None)
        return WsResponse(action=req.action, code=500, msg=msg)

    return WsResponse(
        action=req.action, 
        code=500, 
        msg=f"Knowledge base {req.kb_name} deletion failed."
    )

def update_info(req: KBUpdateInfoRequest) -> WsResponse:
    if not validate_kb_name(req.kb_name):
        return WsResponse(action=req.action, code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(req.kb_name)
    if kb is None:
        return WsResponse(action=req.action, code=404, msg=f"Knowledge base '{req.kb_name}' not found.")
    kb.update_info(req.kb_info)

    return WsResponse()
