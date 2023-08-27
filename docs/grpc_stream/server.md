Module grpc_stream.server
=========================

Classes
-------

`GRPCServe(service: grpc_stream.jobs.Jobs, host_address: str, threads: int = 10)`
:   `GRPCServe` creates a **server** for **gRPC** services.
    
    `GRPCServe` class is used for **hosting** the **gRPC services** for sending data and receive packets of data.
    
    Params:
        `service`: service is expecting one of the **'JOB'** for servicing the **client**.
    
        `host_address`: host_address is expecting the **IP** and **Port** number for setting up the service.
    
        `threads`: threads is the total number of threads the service can handel at the same time. By Default, it
        is `10` threads.

    ### Methods

    `close_line(self)`
    :   `close_line` method is used for closing **gRPC services**.

    `open_line(self)`
    :   `open_line` method is used for starting **gRPC services**, and availing their access.