Module grpc_stream.server
=========================
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