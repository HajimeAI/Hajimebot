import os
import time
import pickle
from pydantic import BaseModel
import asyncio
import aiohttp
import socket
import websockets
import psutil
import platform
import GPUtil
from enum import Enum
from typing import Optional

from configs import (
    logger, WHISPER_MODEL_PATH, WHISPER_MODEL, NODE_INFO_PATH,
    NODE_COMPUTE_CAP_LEVEL, NODE_COMPUTE_CAP_INFO,
    INDEX_TYPE_SELECTION,
)
from whisper import Whisper
from whisper.audio_recorder import AudioRecorder
from communication.manager.solana_util import generate_keypair
from communication.models import *
from model_api.cached_conversation import CachedConversation

class NodeInfo(BaseModel):
    sol_kp: str
    imei: str
    ip: Optional[str] = None
    central_id: Optional[int] = None
    kb_type: Optional[int] = None

def get_n2n_ip():
    # get edge0 AF_INET
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            if interface_name != 'edge0':
                continue
            if str(address.family) == 'AddressFamily.AF_INET':
                return address.address
    return ''

def get_host_ip():
    n2n_ip = get_n2n_ip()
    if n2n_ip:
        return n2n_ip

    try:
        # host_name = socket.gethostname()
        # host_ip = socket.gethostbyname(host_name)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
        s.close()

        return host_ip
    except:
        logger.error("Unable to get Hostname and IP")
        return ''
    
def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

class NodeStatus(Enum):
    BOOTING = 0,
    BOOTED = 1,
    NETWORK_CONNECTING = 2
    NETWORK_CONNECTED = 3,
    WORKING = 4,

class LocalManager():
    def __init__(self):
        self._central_ws = None
        self._status = NodeStatus.BOOTING
        self._conversation = CachedConversation()
        self._index_types = []
        self._connected_ws_set = set()
        self._query_nodes_queue = asyncio.Queue()

    def add_ws(self, ws):
        self._connected_ws_set.add(ws)

    def rmv_ws(self, ws):
        self._connected_ws_set.remove(ws)

    def broadcast_message(self, message):
        websockets.broadcast(self._connected_ws_set, message)

    def set_status(self, status: NodeStatus):
        self._status = status

    @property
    def status(self):
        return self._status
    
    def set_index_types(self, index_types):
        self._index_types = index_types

    def get_index_type_sel(self):
        for index_type in self._index_types:
            if index_type.name == INDEX_TYPE_SELECTION:
                return index_type.id
        return self._node_info.kb_type

    def load_whisper(self):
        self._whisper = Whisper(
            model_path=WHISPER_MODEL_PATH,
            # n_threads=4, 
            # best_of=2,
            # print_progress=False,
            # print_realtime=False,
            # print_colors=True,
        )

    def audio_process_commands(self, raw_data: bytes) -> Tuple[str, str]:
        result = self._whisper.handle_command_workflow(
            raw_data=raw_data, 
            sample_width=AudioRecorder.WIDTH,
            lang='auto', 
        )
        if result is None:
            return None

        lang, txt, prob, elapsed = result
        logger.info(f'lang: {lang}, prob: {prob:.04f}, elapsed {elapsed:.04f} seconds')
        logger.info(f'result: [{txt}]')

        return (lang, txt)

    def load_node_info(self) -> NodeInfo:
        if os.path.isfile(NODE_INFO_PATH):
            with open(NODE_INFO_PATH, 'rb') as fp:
                self._node_info = pickle.load(fp)
        else:
            self.init_node_info()

        logger.info(f'node_info: {self._node_info}')

        return self._node_info
    
    def init_node_info(self):
        sol_kp = generate_keypair()
        self._node_info = NodeInfo(
            sol_kp = str(sol_kp),
            imei = str(sol_kp.pubkey()),
        )
        with open(NODE_INFO_PATH, 'wb') as fp:
            pickle.dump(self._node_info, fp)

    def get_node_info(self):
        return self._node_info

    async def set_login_info(self, login_info: LoginInfoData):
        self._node_info.central_id = login_info.id
        self._node_info.kb_type = login_info.type
        self._node_info.ip = login_info.ip

        n2n_ip = get_n2n_ip()
        if not n2n_ip:
            logger.info(f'n2n_ip not found, exec: {login_info.cmd}')
            await asyncio.to_thread(os.system, login_info.cmd + ' &')
            await self.send_event('p2p_connected', login_info.ip)
        else:
            logger.info(f'n2n_ip found: {n2n_ip}, ignored cmd: {login_info.cmd}')

        await self.send_event('model_online', {
            'text': f'Whisper model({WHISPER_MODEL}) loaded.',
        })
        
        logger.info(f'node_info: {self._node_info}')

    def get_kb_name(self, kb_type=None):
        if not kb_type:
            kb_type = self._node_info.kb_type
        return f'kb_{kb_type}'
    
    def set_central_ws(self, ws):
        self._central_ws = ws
    
    async def send_message_to_central(self, message):
        if not self._central_ws:
            return
        
        await self._central_ws.send(message)

    async def send_event(self, kind, content):        
        node_info = self.get_node_info()
        req = EventRequest(
            msgType='event',
            data=EventData(
                kind=kind,
                node_id=node_info.imei,
                content=content,
            ),
        )
        msg = req.model_dump_json(exclude_none=True)
        logger.info(f'send_event: {msg}')
        await self.send_message_to_central(msg)

    async def send_query(self, type: int):
        node_info = self.get_node_info()
        req = QueryRequest(
            msgType='query',
            data=QueryData(
                imei=node_info.imei,
                type=type,
            ),
        )
        msg = req.model_dump_json(exclude_none=True)
        logger.info(f'send_query: {msg}')
        await self.send_message_to_central(msg)

    async def put_query_node(self, node):
        await self._query_nodes_queue.put(node)

    async def wait_for_query_node(self, kb_type):
        start_tm = time.time()
        while True:
            if self._query_nodes_queue.empty():
                # prevent wait too long
                if time.time() - start_tm >= 5:
                    return None
                await asyncio.sleep(0.1)
                continue
            node = await self._query_nodes_queue.get()
            if node is None:
                return None
            if node.type == kb_type:
                return node
            await self.put_query_node(node)

    async def search_docs_from_neighbor(self, kb_type, query, query_node):
        req = ApiSearchDocsRequest(
            kb_name=self.get_kb_name(kb_type),
            query=query,
        )
        url = f'http://{query_node.ip}:5101/api/search_docs/{self._node_info.imei}'
        logger.info(f'search_docs_from_neighbor: url({url}), kb_name({req.kb_name})')

        payload = req.model_dump(exclude_none=True)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp_json = await resp.text()
                print(f'resp_json: {resp_json}')
                return ApiSearchDocsResponse.model_validate_json(resp_json).chunks

    async def send_log(self, content: str):
        node_info = self.get_node_info()
        req = LogRequest(
            msgType='log',
            data=LogData(
                imei=node_info.imei,
                content=content,
            ),
        )
        await self.send_message_to_central(
            req.model_dump_json(exclude_none=True)
        )

    def collect_node_detail(self):
        node_info = self.get_node_info()

        uname = platform.uname()
        if os.name == 'nt':
            cpu = uname.processor
        else:
            cpu_model = os.popen('lscpu | grep "Model name"').read()
            cpu = cpu_model[cpu_model.index(':') + 1 : ].strip()

        svmem = psutil.virtual_memory()
        partitions = psutil.disk_partitions()
        hd_total = 0
        hd_used = 0
        for partition in partitions:
            try:
                partition_usage = psutil.disk_usage(partition.mountpoint)
                hd_total += partition_usage.total
                hd_used += partition_usage.used
            except PermissionError:
                # this can be catched due to the disk that
                # isn't ready
                continue
        hd_pcnt = '0%'
        if hd_total != 0:
            hd_pcnt = f'{(hd_used * 1000) // hd_total / 10}%'
        
        gpus = GPUtil.getGPUs()
        gpu_name = ','.join([x.name for x in gpus])

        data = NodeDetailData(
            imei=node_info.imei,
            hw_info={
                'cpu': cpu,
                'mem_size': get_size(svmem.total),
                'mem_pcnt': f'{svmem.percent}%',
                'hd_size': get_size(hd_total),
                'hd_pcnt': hd_pcnt,
                'gpu_name': gpu_name,
            },
            compute_cap_level=NODE_COMPUTE_CAP_LEVEL,
            compute_cap=NODE_COMPUTE_CAP_INFO.get(f'level_{NODE_COMPUTE_CAP_LEVEL}', {}),
        )
        
        return NodeDetailRequest(
            msgType='node_detail',
            data=data,
        )

    def simple_chat(self, query):
        logger.info('do simple chat')
        return self._conversation.simple_chat(query)

    def kb_chat(self, query, kb_chunks):
        logger.info('do kb chat')
        reference = '\n'.join(map(lambda x: x['content'].strip(), kb_chunks))
        return self._conversation.kb_chat(query, reference)


g_local_manager = LocalManager()
