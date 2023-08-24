from typing import Generator

import grpc

from grpc_stream import service_pb2
from grpc_stream import service_pb2_grpc


class GRPCConnect:
    def __init__(self, host_address: str, generator: Generator):
        self.address = host_address
        self.gen = generator

    def _client_data_stream(self):
        for row in self.gen:
            service_request = service_pb2.Data(AccV=row.AccV, AccML=row.AccML, AccAP=row.AccAP)
            yield service_request

    def connect_to_stream(self):
        with grpc.insecure_channel(self.address) as channel:
            stub = service_pb2_grpc.PackageStub(channel)
            predictions = stub.bidirectionalStream(self._client_data_stream())
            for pred in predictions:
                print(f'StartHesitation: {pred.StartHesitation}, Turn: {pred.Turn}, Walking: {pred.Walking}')
                # The below code is commented as in this project this method is only used for testing purpose,
                # if you'd like to use the gRPC stream uncomment the below. You should then consider this method as a
                # generator.
                # yield pred
