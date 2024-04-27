import asyncio
import time

from configs import logger
from whisper.audio_recorder import dump_wave_file
from communication.manager.local_manager import g_local_manager
from communication.server_manager import g_server_manager
from communication.api.kb_doc_api import search_docs
from communication.models import *
from model_api.tts import play_audio_stream
from model_api.lang_detect import get_voice_by_lang


async def audio_handle(raw_data: bytes):
    start_tm = time.perf_counter()

    logger.info(f'audio_handle, data len: {len(raw_data)}')
    # dump_wave_file(raw_data)

    result = await asyncio.to_thread(g_local_manager.audio_process_commands, raw_data)
    if result is None:
        return

    lang, transcribe_result = result
    available_langs = {'en', 'ja', 'zh'}
    if lang not in available_langs:
        logger.info(f'lang({lang}) not in {available_langs}, ignored.')
        return
    if not transcribe_result:
        logger.info('Got blank audio, ignored.')
        return

    g_server_manager.put_event('asr', {
        'text': transcribe_result,
    })

    kb_type = g_local_manager.get_index_type_sel()
    kb_rsp = await search_docs_from_kb(SearchDocsFromKBRequest(
        action="search_docs_from_kb",
        kb_type=kb_type,
        query=transcribe_result,
    ))
    kb_chunks = kb_rsp.data
    logger.info(f'kb_chunks: [{kb_chunks}]')
    if not kb_chunks:
        llm_result = await asyncio.to_thread(g_local_manager.simple_chat, transcribe_result)
    else:
        llm_result = await asyncio.to_thread(g_local_manager.kb_chat, transcribe_result, kb_chunks)
    logger.info(f'llm_result: [{llm_result}]')

    g_server_manager.put_event('llm_service', {
        'text': llm_result,
    })

    elapsed = time.perf_counter() - start_tm
    logger.info(f'Before audio playing, elapsed: {elapsed}')

    llm_result_on_line = llm_result.strip().replace("\n", " ")
    voice = get_voice_by_lang(llm_result_on_line)
    await play_audio_stream(llm_result, voice)

    g_server_manager.put_event('tts', {
        'text': 'tts success',
    })


async def search_docs_from_kb(req: SearchDocsFromKBRequest) -> WsResponse:
    node_info = g_local_manager.get_node_info()

    if node_info.kb_type == req.kb_type:
        g_server_manager.put_event('query_vector_db', {
            'from_node': node_info.imei,
            'to_node': node_info.imei,
            'text': req.query,
        })

        # match local kb type
        kb_name = g_local_manager.get_kb_name()
        rsp = await asyncio.to_thread(search_docs, KBSearchDocsRequest(
            action='search_docs',
            query=req.query,
            kb_name=kb_name,
        ))

        chunks = []
        if rsp.code == 0:
            chunks = rsp.data
            log_content = f'search_docs local success: type({req.kb_type}), query({req.query})'
        else:
            log_content = f'search_docs local failed: {rsp.msg}, type({req.kb_type}), query({req.query})'
        logger.info(log_content)
        await g_local_manager.send_log(log_content)

        return WsResponse(
            action=req.action,
            data=chunks,
        )
    else:
        # search from remote node
        await g_local_manager.send_query(req.kb_type)

        query_node = await g_local_manager.wait_for_query_node(req.kb_type)
        if query_node is None:
            logger.info(f'No query_node for {req.kb_type}')
            return WsResponse(
                action=req.action,
                data=[],
            )

        logger.info(f'Got remote query_node: {query_node}')
        g_server_manager.put_event('query_vector_db', {
            'from_node': node_info.imei,
            'to_node': query_node.imei,
            'text': req.query,
        })

        chunks = await g_local_manager.search_docs_from_neighbor(
            kb_type=req.kb_type,
            query=req.query,
            query_node=query_node,
        )

        await g_local_manager.send_log(
            f'search_docs from remote({query_node}) success: type({req.kb_type}), query({req.query})')

        return WsResponse(
            action=req.action,
            data=chunks,
        )
