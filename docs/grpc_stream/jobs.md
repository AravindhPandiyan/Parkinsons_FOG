Module grpc_stream.jobs
=======================

Classes
-------

`Jobs()`
:   `Jobs` is an empty class used for **grouping classes** of similar type under one. This class is **inherited**
by the similar classes to become part of the same **group**. The work of any **'JOB'** class is for setting
up services like **Unary RPC, Server-side streaming RPC, Client-side streaming RPC and Bi-directional streaming
RPC**. The 'JOB' classes will be specific to their work. There can exist multiple **'JOB'** of same **service**.

    ### Descendants

    * grpc_stream.jobs.PredictorJob

    ### Class variables

    `NAME`
    :

`PredictorJob(inference)`
:   `PredictorJob` is a **service** for the servicing the **client connections**.

    `PredictorJob` class is part of the main **'JOB'** group of classes. This **'JOB'** is used for
    **Bi-directional streaming** of data and predictions between **client** and **server** respectively. This
    class also **inherits the Servicer** class from `pb2_grpc.py` file.
    
    Params:
        `inference`: inference is expecting the inference class of the model.

    ### Ancestors (in MRO)

    * grpc_stream.jobs.Jobs
    * grpc_stream.service_pb2_grpc.PackageServicer

    ### Class variables

    `CHILD_NAME`
    :

    ### Methods

    `bidirectionalStream(self, request_iterator, context)`
    :   `bidirectionalStream` method is used to **override** the method in the servicer class and allows to
        write down the specific code for handling **Bi-directional streaming** of the data packages.
        
        Params:
            `request_iterator`: request_iterator is used for **iterating** over the data packets sent form
            the client side.
        
            `context`: context contains the **metadata** of the **client**, method name and deadline for the calls**.
        
        Returns:
            This method **streams predictions to the client**. It acts like a **Generator**.