from concurrent import futures

import grpc

from grpc_stream import service_pb2_grpc
from grpc_stream.jobs import Jobs


class GRPCServe:
    """
    GRPCServe class is used for hosting the gRPC services for sending data and receive packets of data.
    """

    def __init__(self, service: Jobs, host_address: str, threads: int = 10):
        """
        GRPCServe class is used for hosting the gRPC services for sending data and receive packets of data.
        :param service: service is expecting one of the 'JOB' for servicing the client.
        :param host_address: host_address is expecting the IP and Port number for setting up the service.
        :param threads: threads is the total number of threads the service can handel at the same time. By Default, it
        is 10 threads.
        """
        self.services = service
        self.address = host_address
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=threads))

    def open_line(self):
        """
        open_line method is used for starting gRPC services, and availing their access.
        """
        service_pb2_grpc.add_PackageServicer_to_server(self.services, self._server)
        self._server.add_insecure_port(self.address)
        self._server.start()
        self._server.wait_for_termination(timeout=60.0)

    def close_line(self):
        """
        close_line method is used for closing gRPC services.
        """
        self._server.stop()
