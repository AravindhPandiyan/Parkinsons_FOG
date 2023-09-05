Module api.app.models.inference_model
=====================================

Classes
-------

`GRPCResponseModel(**data: Any)`
:   `GRPCResponseModel` is a validation class used for checking the **datatype** and **structuring** the output
of an API.

    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises ValidationError if the input data cannot be parsed to form a valid model.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel
    * pydantic.utils.Representation

    ### Descendants

    * pydantic.main.GRPCResponseModel

    ### Class variables

    `gRPC_stream_address: str`
    :

`Options(value, names=None, *, module=None, qualname=None, type=None, start=1)`
:   `Options` is an Enum class that provides the options for **Bi-directional Streaming** of data that the user
can select during API request.

    ### Ancestors (in MRO)

    * builtins.str
    * enum.Enum

    ### Class variables

    `RPC`
    :

    `WS`
    :

`SocketPackageModel(**data: Any)`
:   `SocketPackageModel` is a validation class for checking the **datatype** and structure of the data received from
the user in an API call. This class is used for the checking the data packet received on the Web-Socker.

    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises ValidationError if the input data cannot be parsed to form a valid model.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel
    * pydantic.utils.Representation

    ### Class variables

    `AccAP: float`
    :

    `AccML: float`
    :

    `AccV: float`
    :

`StreamingOptionModel(**data: Any)`
:   `StreamingOptionModel` is a validation class that uses the above Enums for validating the **datatype**
and **structure** of the data received from the user in an API call. This Class is used for providing the
user to choose between the **Web-Socket** and **gRPC** for **streaming** data.

    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises ValidationError if the input data cannot be parsed to form a valid model.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel
    * pydantic.utils.Representation

    ### Class variables

    `option: api.app.models.inference_model.Options`
    :

`WebSocketResponseModel(**data: Any)`
:   `WebSocketResponseModel` is a validation class used for checking the **datatype** and **structuring** the
output of an API.

    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises ValidationError if the input data cannot be parsed to form a valid model.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel
    * pydantic.utils.Representation

    ### Descendants

    * pydantic.main.WebSocketResponseModel

    ### Class variables

    `web_socket_address: str`
    :