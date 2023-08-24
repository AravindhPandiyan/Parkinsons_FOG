from concurrent import futures

import grpc

from grpc_stream import service_pb2_grpc
from grpc_stream.jobs import Jobs


class GRPCServe:
    def __init__(self, service: Jobs, host_address: str, threads: int = 10):
        self.services = service
        self.address = host_address
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=threads))

    def open_line(self):
        service_pb2_grpc.add_PackageServicer_to_server(self.services, self._server)
        self._server.add_insecure_port(self.address)
        self._server.start()
        self._server.wait_for_termination()

    def close_line(self):
        self._server.stop()
