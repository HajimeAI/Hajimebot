import uvicorn
import signal
import queue

from communication.manager.local_threads import LocalWorkerThread

# FastAPI Uvicorn override
class RestfulServer(uvicorn.Server):

    # Override
    def install_signal_handlers(self) -> None:
        # Do nothing
        pass

class ServerManager():
    def __init__(self):
        self._restful_servers = []
        self._ws_servers = []
        self._is_running = True
        self._task_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._event_queue = queue.Queue()
        signal.signal(signal.SIGINT, lambda _, __: self.terminate_all())

    def start_local_worker_thread(self):
        self._local_worker = LocalWorkerThread(
            task_queue=self._task_queue, 
            response_queue=self._response_queue,
        )
        self._local_worker.start()
    
    def add_local_task(self, func, kwargs):
        self._task_queue.put((func, kwargs))
    
    def get_local_response(self):
        if self._response_queue.empty():
            return None
        return self._response_queue.get()
    
    def put_event(self, kind, content):
        self._event_queue.put((kind, content))
    
    def get_event(self):
        if self._event_queue.empty():
            return None
        return self._event_queue.get()

    def reg_ws_server(self, server):
        self._ws_servers.append(server)

    def create_restful_server(self, config: uvicorn.Config):
        server = RestfulServer(config)
        self._restful_servers.append(server)
        return server

    def terminate_all(self):
        for svr in self._ws_servers:
            svr.close()
        for svr in self._restful_servers:
            svr.should_exit = True

        self._local_worker.stop()
        self._local_worker.join()
        self._is_running = False
    
    def is_running(self):
        return self._is_running

g_server_manager = ServerManager()
