"""
gRPC Server Setup
-----------------

This module defines the `GRPCServe` class, which is used to create and manage a gRPC server for hosting gRPC services.
The server can host various types of gRPC services (defined by 'JOB' classes) that allow communication between clients
and the server using different types of gRPC interactions.

Classes:
- `GRPCServe`: A class used to create a gRPC server and host specified gRPC services.

Usage:
- Import the necessary modules and classes.
- Create an instance of the `Jobs`-derived 'JOB' class that corresponds to the gRPC service you want to host.
- Create an instance of the `GRPCServe` class, passing the 'JOB' class instance, the host IP and port, and optionally
  the number of threads to use for handling requests.
- Use the `open_line()` method to start the gRPC services and make them available for communication.
- After you're done, use the `close_line()` method to stop the gRPC services and close the server.
"""

from concurrent import futures

import grpc

from grpc_stream import service_pb2_grpc
from grpc_stream.jobs import Jobs


class GRPCServe:
    """
    `GRPCServe` creates a **server** for **gRPC** services.
    """

    def __init__(self, service: Jobs, host_address: str, threads: int = 10):
        """
        `GRPCServe` class is used for **hosting** the **gRPC services** for sending data and receive packets of data.

        Params:
            `service`: service is expecting one of the **'JOB'** for servicing the **client**.

            `host_address`: host_address is expecting the **IP** and **Port** number for setting up the service.

            `threads`: threads is the total number of threads the service can handel at the same time. By Default, it
            is `10` threads.
        """
        self.services = service
        self.address = host_address
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=threads))

    def open_line(self):
        """
        `open_line` method is used for starting **gRPC services**, and availing their access.
        """
        service_pb2_grpc.add_PackageServicer_to_server(self.services, self._server)
        self._server.add_insecure_port(self.address)
        self._server.start()
        self._server.wait_for_termination(timeout=60.0)

    def close_line(self):
        """
        `close_line` method is used for closing **gRPC services**.
        """
        self._server.stop()
