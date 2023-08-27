Module grpc_stream.client
=========================

Classes
-------

`GRPCConnect(data_generator: Generator, host_address: str, state: int = 1)`
:   `GRPCConnect` connects to the **gRPC** server **servicer**.
    
    `GRPCConnect` class is used for connecting to the **gRPC stream** to send data and receive packets of data.
    
    Params:
        `data_generator`: data_generator is the **source** to send individual packets of data to the server.
    
        `host_address`: host_address is the **IP** and **Port** to which the client must connect to access
        the stream.
    
        `state`: **{0 or ‘Test’, 1 or ‘Generator’}**, default `1`. This is for setting if the connect_to_stream
        method should act as normal **method** or return a **generator**.

    ### Methods

    `connect_to_stream(self) ‑> Generator[dict] | None`
    :   `connect_to_stream` connects to the **gRPC streaming channel** and start the **full-duplex streaming**.
        This method has `2` **states; Method state**, where it will keep printing the data, this state is
        only for testing purpose. **Generator state**, where it will allow the user to receive the data continuously.
        
        Returns:
            This method will return a **generator** if the state is `1` or above else it will act like a normal method.