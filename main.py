import asyncio

from communication.local import local_ws_server, local_restful_server
from communication.central import start_central_connection, central_responsing_loop
from communication.node2_neighbor import node2_neighbor_server

async def main():
    tasks = [
        local_ws_server(),
        local_restful_server(),
        start_central_connection(),
        central_responsing_loop(),
        node2_neighbor_server(),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
