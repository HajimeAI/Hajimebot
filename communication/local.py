import inspect
import websockets
import json
import asyncio
from fastapi import FastAPI
import uvicorn

from configs import logger, NLTK_DATA_PATH, WS_MAX_SIZE
from communication.models import *
import communication.api as api
from communication.manager.local_manager import g_local_manager
from communication.server_manager import g_server_manager

import nltk

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


async def search_docs_from_kb(req: SearchDocsFromKBRequest):
    return await api.search_docs_from_kb(req)


async def list_kbs(req: WsBase):
    return await asyncio.to_thread(api.list_kbs, req)


async def create_kb(req: KBCreateRequest):
    return await asyncio.to_thread(api.create_kb, req)


async def delete_kb(req: KBDeleteRequest):
    return await asyncio.to_thread(api.delete_kb, req)


async def update_info(req: KBUpdateInfoRequest):
    return await asyncio.to_thread(api.update_info, req)


async def list_files(req: KBListFilesRequest):
    return await asyncio.to_thread(api.list_files, req)


async def update_docs(req: KBUpdateDocsRequest):
    return await asyncio.to_thread(api.update_docs, req)


async def delete_docs(req: KBDeleteDocsRequest):
    return await asyncio.to_thread(api.delete_docs, req)


async def recreate_vector_store(req: KBRecreateVectorStoreRequest):
    for result in api.recreate_vector_store(req):
        yield result


async def search_docs(req: KBSearchDocsRequest):
    return await asyncio.to_thread(api.search_docs, req)


action_list = [
    (search_docs_from_kb, SearchDocsFromKBRequest),
    (list_kbs, WsBase),
    (create_kb, KBCreateRequest),
    (delete_kb, KBDeleteRequest),
    (update_info, KBUpdateInfoRequest),
    (list_files, KBListFilesRequest),
    (update_docs, KBUpdateDocsRequest),
    (delete_docs, KBDeleteDocsRequest),
    (recreate_vector_store, KBRecreateVectorStoreRequest),
    (search_docs, KBSearchDocsRequest),
]
actions_map = {}
for x in action_list:
    actions_map[x[0].__name__] = x


async def on_connected(websocket: websockets.WebSocketServerProtocol):
    try:
        # Register new ws connection
        g_local_manager.add_ws(websocket)

        # Send init messages

        # Manage state changes
        async for message in websocket:
            try:
                if type(message) == bytes:
                    try:
                        await api.audio_handle(message)
                    except Exception as e:
                        logger.error(f'audio_handle: {e}', exc_info=e)
                    finally:
                        # require audio record resuming
                        await websocket.send(
                            WsBase(action='record_resume').model_dump_json(exclude_none=True)
                        )
                    continue

                event = json.loads(message)
                reqBase = WsBase.model_validate(event)
                if reqBase.action not in actions_map:
                    logger.error(f'Invalid action: {reqBase.action}')
                    continue

                func, req = actions_map[reqBase.action]

                if inspect.isasyncgenfunction(func):
                    gen = func(req.model_validate(event))
                    async for result in gen:
                        await websocket.send(
                            result.model_dump_json(exclude_none=True)
                        )
                else:
                    result = await func(req.model_validate(event))
                    if result:
                        await websocket.send(
                            result.model_dump_json(exclude_none=True)
                        )
            except Exception as error:
                logger.error(f"Message handle error: {error}", exc_info=error)
    finally:
        # Remove connection
        g_local_manager.rmv_ws(websocket)

        # Broacast to other connections


async def local_ws_server():
    g_local_manager.load_models()

    server = await websockets.serve(
        on_connected,
        "0.0.0.0", 5001,
        max_size=WS_MAX_SIZE,
        start_serving=True,
    )
    g_server_manager.reg_ws_server(server)
    # await server.start_serving()
    await server.wait_closed()
    # await asyncio.Future()  # run forever
    logger.info('local_ws_server exit')


app = FastAPI()


@app.post("/api/download_doc")
async def api_download_doc(body: KBDownloadDocRequest):
    return api.download_doc(body)


async def local_restful_server():
    config = uvicorn.Config("communication.local:app", host='0.0.0.0', port=5002, log_level="info")
    server = g_server_manager.create_restful_server(config)
    await server.serve()
    logger.info('local_restful_server exit')
