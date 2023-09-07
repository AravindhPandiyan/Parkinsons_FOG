# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import grpc_stream.rpc_service_pb2 as rpc__service__pb2


class PackageStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.bidirectionalStream = channel.stream_stream(
            "/biStream.Package/bidirectionalStream",
            request_serializer=rpc__service__pb2.Data.SerializeToString,
            response_deserializer=rpc__service__pb2.Prediction.FromString,
        )


class PackageServicer(object):
    """Missing associated documentation comment in .proto file."""

    def bidirectionalStream(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_PackageServicer_to_server(servicer, server):
    """
    add_PackageServicer_to_server method is used to add The Jobs to be serviced in the server side for clients'
    connections to make use.
    """
    rpc_method_handlers = {
        "bidirectionalStream": grpc.stream_stream_rpc_method_handler(
            servicer.bidirectionalStream,
            request_deserializer=rpc__service__pb2.Data.FromString,
            response_serializer=rpc__service__pb2.Prediction.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "biStream.Package", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class Package(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def bidirectionalStream(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        """
        bidirectionalStream is method used for setting up the Bi-directional Streaming process.
        """
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/biStream.Package/bidirectionalStream",
            rpc__service__pb2.Data.SerializeToString,
            rpc__service__pb2.Prediction.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
