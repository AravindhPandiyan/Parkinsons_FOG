Module grpc_stream.service_pb2_grpc
===================================
Client and server classes corresponding to protobuf-defined services.

Functions
---------

    
`add_PackageServicer_to_server(servicer, server)`
:   This method adds a gRPC service to a server

Classes
-------

`Package()`
:   Missing associated documentation comment in .proto file.

    ### Static methods

    `bidirectionalStream(request_iterator, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None)`
    :

`PackageServicer()`
:   Missing associated documentation comment in .proto file.

    ### Descendants

    * grpc_stream.jobs.PredictorJob

    ### Methods

    `bidirectionalStream(self, request_iterator, context)`
    :   Missing associated documentation comment in .proto file.

`PackageStub(channel)`
:   Missing associated documentation comment in .proto file.
    
    Constructor.
    
    Params:
        channel: A grpc.Channel.