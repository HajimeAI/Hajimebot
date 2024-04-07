import os
import json
import time
import random
import websockets
import asyncio
from urllib.parse import urlencode

from configs import (
    logger, CENTRAL_SERVER_WS_URL, EMBEDDING_MODEL_PATH, WS_MAX_SIZE
)
from communication.manager.local_manager import g_local_manager, NodeStatus
from communication.server_manager import g_server_manager
import communication.api as api
from communication.models import *
from model_api.tts import create_const_speech


async def handle_login(req: LoginInfo):
    logger.info(f'login: {req}')

    await g_local_manager.set_login_info(req.data)

    if os.path.exists(EMBEDDING_MODEL_PATH):
        kb_name = g_local_manager.get_kb_name()
        rsp = await asyncio.to_thread(api.create_kb, KBCreateRequest(
            action='create_kb',
            kb_name=kb_name,
        ))
        if rsp.code == 0:
            await g_local_manager.send_log(f'Knowledge base creation success: {kb_name}')
        elif rsp.code > 0:
            logger.error(f'create_kb error: {rsp.msg}')
    
    if g_local_manager.status == NodeStatus.NETWORK_CONNECTING:
        g_local_manager.set_status(NodeStatus.NETWORK_CONNECTED)
    return g_local_manager.collect_node_detail()

async def handle_query_indexed(req: QueryIndexed):
    logger.info(f'handle_query_indexed: {req}')
    g_local_manager.set_index_types(req.data)

async def handle_ping(req: HeartBeat):
    req.msgType = 'pong'
    return req

async def handle_file(req: FileTransfer):
    def _inner_func(req: FileTransfer) -> FileTransferResponse:
        data = req.data
        file_name = data.fileName
        kb_name = g_local_manager.get_kb_name()

        try:
            rsp = api.upload_file(file_name, kb_name)
            if rsp.code != 0:
                logger.error(f'handle_file({kb_name}, {file_name}) error: {rsp.msg}')
            else:
                logger.info(f'handle_file({kb_name}, {file_name}) success.')
            
            return FileTransferResponse(
                msgType=req.msgType,
                data=CentralWsRspDataBase(
                    code=rsp.code,
                    msg=rsp.msg,
                ),
            )
            
        except Exception as e:
            msg = f'handle_file: {e}'
            logger.error(msg, exc_info=e)
            return FileTransferResponse(
                msgType=req.msgType,
                data=CentralWsRspDataBase(
                    code=500,
                    msg=msg,
                ),
            )
    
    g_server_manager.add_local_task(_inner_func, {'req': req})

async def handle_file_del(req: FileDel):
    def _inner_func(req: FileDel) -> FileDelResponse:
        data = req.data
        file_name = data.fileName
        kb_name = g_local_manager.get_kb_name()

        try:
            rsp = api.delete_docs(KBDeleteDocsRequest(
                action='delete_docs',
                kb_name=kb_name,
                file_names=[file_name],
                delete_content=True,
            ))
            if rsp.code != 0:
                logger.error(f'handle_file_del({kb_name}, {file_name}) error: {rsp.msg}')
            else:
                logger.info(f'handle_file_del({kb_name}, {file_name}) success.')
                g_server_manager.put_event('file_embedding', {
                    'text': file_name,
                })
            
            return FileDelResponse(
                msgType=req.msgType,
                data=CentralWsRspDataBase(
                    code=rsp.code,
                    msg=rsp.msg,
                ),
            )
            
        except Exception as e:
            msg = f'handle_file_del: {e}'
            logger.error(msg, exc_info=e)
            return FileDelResponse(
                msgType=req.msgType,
                data=CentralWsRspDataBase(
                    code=500,
                    msg=msg,
                ),
            )
    
    g_server_manager.add_local_task(_inner_func, {'req': req})

async def handle_query(req: QueryResponse):
    logger.info(f'handle_query: {req}')
    if len(req.data) == 0:
        await g_local_manager.put_query_node(None)
    else:
        await g_local_manager.put_query_node(random.choice(req.data))

###########################################################

action_list = [
    (handle_login, LoginInfo),
    (handle_query_indexed, QueryIndexed),
    (handle_ping, HeartBeat),
    (handle_file, FileTransfer),
    (handle_file_del, FileDel),
    (handle_query, QueryResponse),
]
actions_map = {}
for x in action_list:
    actions_map[x[0].__name__[len('handle_'):]] = x

###########################################################

async def start_central_connection():
    g_server_manager.start_local_worker_thread()
    await create_const_speech()
    g_local_manager.set_status(NodeStatus.BOOTED)

    node_info = g_local_manager.load_node_info()
    params = {
        'imei': node_info.imei,
    }
    url = CENTRAL_SERVER_WS_URL + '?' + urlencode(params)

    while True:
        if g_local_manager.status == NodeStatus.BOOTED:
            await asyncio.sleep(0.1)
            continue
        
        try:
            logger.info(f'connecting central ws: {url}')
            websocket = await websockets.connect(url, max_size=WS_MAX_SIZE)
            g_local_manager.set_central_ws(websocket)

            try:
                async for message in websocket:
                    # print(f'recv message: {message}')
                    event = json.loads(message)
                    reqBase = CentralWsBase.model_validate(event)
                    if (not reqBase.msgType) or (reqBase.msgType not in actions_map):
                        logger.error(f'Invalid message type: {reqBase.msgType}')
                        continue

                    func, req = actions_map[reqBase.msgType]
                    result = await func(req.model_validate(event))
                    if result:
                        await websocket.send(
                            result.model_dump_json(exclude_none=True)
                        )
                    if not g_server_manager.is_running():
                        break
                
                logger.info(f'central websocket client is closing ...')
                await websocket.close()
                logger.info(f'central websocket client closed.')
                if not g_server_manager.is_running():
                    break

            except websockets.ConnectionClosed as e:
                logger.error(f'websocket error: {e}, reconnect...')
                continue
        except Exception as e:
            logger.error(f'websocket connect error: {e}, reconnect...', exc_info=e)
            time.sleep(3)
            continue

async def central_responsing_loop():
    while g_server_manager.is_running():
        working = False

        rsp = g_server_manager.get_local_response()
        if rsp:
            msg = rsp.model_dump_json(exclude_none=True)
            logger.info(f'central response: {msg}')
            await g_local_manager.send_message_to_central(msg)
            working = True

        ev = g_server_manager.get_event()
        if ev:
            kind, content = ev
            await g_local_manager.send_event(kind, content)
            working = True

        if not working:
            await asyncio.sleep(0.1)
        
        


