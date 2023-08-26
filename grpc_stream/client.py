from __future__ import annotations

from typing import Generator

import grpc

from grpc_stream import service_pb2, service_pb2_grpc


class GRPCConnect:
    """
    GRPCConnect class is used for connecting to the gRPC stream to send data and receive packets of data.
    """

    def __init__(self, data_generator: Generator, host_address: str, state: int = 1):
        """
        GRPCConnect class is used for connecting to the gRPC stream to send data and receive packets of data.
        :param host_address: host_address is the IP and Port to which the client must connect to access the stream.
        :param data_generator: data_generator is the source to send individual packets of data to the server.
        :param state: {0 or ‘Test’, 1 or ‘Generator’}, default 1. This is for setting if the connect_to_stream method
        should act as normal method or return a generator.
        """
        self.address = host_address
        self.gen = data_generator
        self.method_state = state

    def _client_data_stream(self) -> Generator:
        """
        _client_data_stream is a private generator method used for iterating over the provide data generator to send
        individual packets of data.
        :return: Every time the method is called the next data packet in the generator is returned.
        """
        for row in self.gen:
            service_request = service_pb2.Data(
                AccV=row.AccV, AccML=row.AccML, AccAP=row.AccAP
            )
            yield service_request

    def connect_to_stream(self) -> Generator[dict] | None:
        """
        connect_to_stream connects to the gRPC streaming channel and start the full-duplex streaming. This method has 2
        states; Method state, where it will keep printing the data, this state is only for testing purpose.
        Generator state, where it will allow the user to receive the data continuously.
        :return: This method will return a generator if the state is 1 else it will act like a normal method.
        """
        with grpc.insecure_channel(self.address) as channel:
            stub = service_pb2_grpc.PackageStub(channel)
            predictions = stub.bidirectionalStream(self._client_data_stream())

            if self.method_state:
                return (pred for pred in predictions)
            else:
                for pred in predictions:
                    print(
                        f"StartHesitation: {pred.StartHesitation}, Turn: {pred.Turn}, Walking: {pred.Walking}"
                    )
