import os
import threading
import queue
import time

from configs import logger, CONST_SPEECH_MAP, SPEECH_DELAY_SECONDS
from communication.manager.local_manager import g_local_manager, NodeStatus
from model_api.tts import play_const_audio

class LocalWorkerThread(threading.Thread):
    def __init__(self, task_queue: queue.Queue, response_queue: queue.Queue):
        super().__init__()
        self._running: bool = True
        self._task_queue = task_queue
        self._response_queue = response_queue
        self._booted_time = time.time()

    def run(self):
        while self._running:
            if g_local_manager.status == NodeStatus.BOOTED:
                speech = CONST_SPEECH_MAP['BOOTED']
                if not os.path.exists(speech['file']):
                    time.sleep(0.1)
                    continue
                else:
                    play_const_audio('BOOTED')
                    self._booted_time = time.time()
                    g_local_manager.set_status(NodeStatus.NETWORK_CONNECTING)
                    continue
            elif g_local_manager.status == NodeStatus.NETWORK_CONNECTED:
                now = time.time()
                speech = CONST_SPEECH_MAP['CONNECTED']
                if ((not os.path.exists(speech['file'])) or 
                    now < self._booted_time + SPEECH_DELAY_SECONDS):
                    
                    time.sleep(0.1)
                    continue
                else:
                    play_const_audio('CONNECTED', block=False)
                    g_local_manager.set_status(NodeStatus.WORKING)
                    continue

            if g_local_manager.status != NodeStatus.WORKING or self._task_queue.empty():
                time.sleep(0.1)
                continue

            func, kwargs = self._task_queue.get()
            result = func(**kwargs)
            if result:
                self._response_queue.put(result)
        
        logger.info('LocalWorkerThread exit.')

    def stop(self):
        self._running = False
