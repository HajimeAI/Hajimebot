import urllib
import os
from fastapi.responses import FileResponse
import requests

from configs import logger, log_verbose, CENTRAL_SERVER_DOWNLOAD_URL, CENTRAL_SERVER_PROXIES
from communication.models import *
from knowledge_base.utils import validate_kb_name, list_files_from_folder, get_file_path
from knowledge_base.kb_service.base import KBServiceFactory
from knowledge_base.knowledge_file import KnowledgeFile, files2docs_in_thread

def list_files(
    req: KBListFilesRequest
) -> WsResponse:
    """
    listing files in knowledge base
    """
    if not validate_kb_name(req.kb_name):
        return WsResponse(action=req.action, code=403, msg="Don't attack me")

    req.kb_name = urllib.parse.unquote(req.kb_name)
    kb = KBServiceFactory.get_service_by_name(req.kb_name)
    if kb is None:
        return WsResponse(action=req.action, code=404, msg=f"Knowledge base '{req.kb_name}' not found.")
    else:
        all_doc_names = kb.list_files()
        return WsResponse(
            action=req.action, 
            data=KBListFilesData(
                data=all_doc_names,
            )
        )

def upload_file(file_name: str, kb_name: str) -> WsResponse:
    logger.info(f'upload_file: file_name({file_name}), kb_name({kb_name}))')
    file_path = get_file_path(knowledge_base_name=kb_name, doc_name=file_name)

    url = f'{CENTRAL_SERVER_DOWNLOAD_URL}/{file_name}'
    logger.info(f'get file from: {url}')
    with requests.get(url, stream=True, proxies=CENTRAL_SERVER_PROXIES) as r:
        if not r.ok:
            return WsResponse(action='upload_file', code=403, msg=f"Download {file_name} failed: {r.status_code}") 
        
        with open(file_path, 'wb') as fp:
            for chunk in r.iter_content(chunk_size=8192):
                fp.write(chunk)
            
    logger.info(f'get file done: {url}')

    kb = KBServiceFactory.get_service_by_name(kb_name)
    rsp = update_docs(
        KBUpdateDocsRequest(
            action='upload_file',
            kb_name=kb_name,
            file_names=[file_name],
            not_refresh_vs_cache=True,
        ),
    )
    
    kb.save_vector_store()

    logger.info(f'upload_file: file_name({file_name}), kb_name({kb_name})) finished.')

    return rsp

def update_docs(req: KBUpdateDocsRequest) -> WsResponse:
    """
    update file in knowledge base
    """
    if not validate_kb_name(req.kb_name):
        return WsResponse(action=req.action, code=403, msg="Don't attack me")

    req.kb_name = urllib.parse.unquote(req.kb_name)
    kb = KBServiceFactory.get_service_by_name(req.kb_name)
    if kb is None:
        return WsResponse(action=req.action, code=404, msg=f"Knowledge base '{req.kb_name}' not found.")

    failed_files = {}
    kb_files = []

    # generate docs to load
    for file_name in req.file_names:
        try:
            kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=req.kb_name))
        except Exception as e:
            msg = f"'{file_name}' loading error: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                            exc_info=e if log_verbose else None)
            failed_files[file_name] = msg
            

    # from file to docs and vectorize
    for status, result in files2docs_in_thread(
        kb_files,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        zh_title_enhance=req.zh_title_enhance
    ):
        if status:
            _, file_name, new_docs = result
            kb_file = KnowledgeFile(
                filename=file_name,
                knowledge_base_name=req.kb_name,
            )
            kb_file.splited_docs = new_docs
            kb.update_doc(kb_file, not_refresh_vs_cache=True)
        else:
            _, file_name, error = result
            failed_files[file_name] = error

    if not req.not_refresh_vs_cache:
        kb.save_vector_store()

    return WsResponse(
        action=req.action, 
        data=KBUpdateDocsData(
            failed_files=failed_files,
        )
    )

def delete_docs(req: KBDeleteDocsRequest) -> WsResponse:
    """
    delete docs in kownledge base
    """
    if not validate_kb_name(req.kb_name):
        return WsResponse(action=req.action, code=403, msg="Don't attack me")

    req.kb_name = urllib.parse.unquote(req.kb_name)
    kb = KBServiceFactory.get_service_by_name(req.kb_name)
    if kb is None:
        return WsResponse(action=req.action, code=404, msg=f"Knowledge base '{req.kb_name}' not found.")

    failed_files = {}
    for file_name in req.file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"'{file_name}' not found."

        try:
            kb_file = KnowledgeFile(
                filename=file_name,
                knowledge_base_name=req.kb_name
            )
            kb.delete_doc(kb_file, req.delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"{file_name} deletion error: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not req.not_refresh_vs_cache:
        kb.save_vector_store()

    return WsResponse(
        action=req.action, 
        data=KBDeleteDocsData(
            failed_files=failed_files,
        )
    )

def recreate_vector_store(req: KBRecreateVectorStoreRequest):
    """
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
    """
    if not validate_kb_name(req.kb_name):
        yield WsResponse(action=req.action, code=403, msg="Don't attack me")
        return

    req.kb_name = urllib.parse.unquote(req.kb_name)
    kb = KBServiceFactory.get_service(req.kb_name, req.vs_type, req.embed_model)
    kb_exists = kb.exists()
    if not kb_exists:
        yield WsResponse(action=req.action, code=404, msg=f"Knowledge base '{req.kb_name}' not found.")
        return
    
    if kb.exists():
        kb.clear_vs()
    kb.create_kb()
    files = list_files_from_folder(req.kb_name)
    kb_files = [(file, req.kb_name) for file in files]

    i = 0
    files_cnt = len(files)
    for status, result in files2docs_in_thread(
        kb_files,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        zh_title_enhance=req.zh_title_enhance
    ):
        if status:
            kb_name, file_name, docs = result
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
            kb_file.splited_docs = docs
            kb.add_doc(kb_file, not_refresh_vs_cache=True)

            yield WsResponse(
                action=req.action, 
                data=KBRecreateVectorStoreData(
                    total=files_cnt,
                    finished=(i + 1),
                    doc=file_name,
                )
            )
        else:
            kb_name, file_name, error = result
            msg = f"Error occured when adding '{file_name}' to '{kb_name}': {error}, skip it."
            logger.error(msg)
            yield WsResponse(action=req.action, code=500, msg=msg)
        i += 1
    
    if not req.not_refresh_vs_cache:
        kb.save_vector_store()

def search_docs(req: KBSearchDocsRequest):
    if not validate_kb_name(req.kb_name):
        return WsResponse(action=req.action, code=403, msg="Don't attack me")

    req.kb_name = urllib.parse.unquote(req.kb_name)
    
    return WsResponse(
        action=req.action, 
        data=inner_search_docs(req)
    )

def inner_search_docs(req: KBSearchDocsRequest):
    kb = KBServiceFactory.get_service_by_name(req.kb_name)
    data = []
    if kb is not None:
        if req.query:
            docs = kb.search_docs(req.query, req.top_k, req.score_threshold)
            data = [
                {
                    'id': doc.metadata.get("id"),
                    'content': doc.page_content,
                    'score': float(score),
                } for doc, score in docs
            ]
        elif req.file_name or req.metadata:
            docs = kb.list_docs(file_name=req.file_name, metadata=req.metadata)
            for d in docs:
                if "vector" in d.metadata:
                    del d.metadata["vector"]
            data = [
                {
                    'id': doc.id,
                    'content': doc.page_content,
                    'score': doc.score,
                } for doc in docs
            ]
    
    return data

###########################################################

def download_doc(req: KBDownloadDocRequest):
    """
    download docs in knowledge base
    """
    if not validate_kb_name(req.kb_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(req.kb_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Knowledge base '{req.kb_name}' not found.")

    if req.preview:
        content_disposition_type = "inline"
    else:
        content_disposition_type = None

    try:
        kb_file = KnowledgeFile(
            filename=req.file_name,
            knowledge_base_name=req.kb_name,
        )

        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"'{kb_file.filename}' reading failed: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"'{kb_file.filename}' reading failed.")
