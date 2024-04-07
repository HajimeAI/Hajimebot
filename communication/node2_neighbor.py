from fastapi import FastAPI
import uvicorn

from configs import logger
from communication.api import *
from communication.models import *
from communication.server_manager import g_server_manager
from communication.manager.local_manager import g_local_manager

app = FastAPI()

@app.post("/api/search_docs/{imei}")
async def api_search_docs(imei: str, body: ApiSearchDocsRequest) -> ApiSearchDocsResponse:
    logger.info(f'search_docs(from {imei}) starting ...')
    rsp = search_docs(KBSearchDocsRequest(
        action='search_docs',
        query=body.query,
        kb_name=body.kb_name,
    ))

    chunks = []
    if rsp.code == 0:
        chunks = rsp.data
        log_content = f'search_docs(from {imei}) success: {body}'
    else:
        log_content = f'search_docs(from {imei}) failed: {rsp.msg}, {body}'
    logger.info(log_content)
    await g_local_manager.send_log(log_content)
    g_server_manager.put_event('vector_db_response', {
        'from_node': imei,
        'to_node': g_local_manager.get_node_info().imei,
        'text': body.query,
    })

    return ApiSearchDocsResponse(
        chunks=chunks,
    )

async def node2_neighbor_server():
    config = uvicorn.Config("communication.node2_neighbor:app", host='0.0.0.0', port=5101, log_level="info")
    server = g_server_manager.create_restful_server(config)
    await server.serve()
    logger.info('node2_neighbor_server exit')
