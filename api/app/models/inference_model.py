from enum import Enum

from pydantic import BaseModel


class Options(str, Enum):
    """
    `Options` is an Enum class that provides the options for **Bi-directional Streaming** of data that the user
    can select during API request.
    """

    WS = "WebSocket"
    RPC = "gRPC"


class WebSocketResponseModel(BaseModel):
    """
    `WebSocketResponseModel` is a validation class used for checking the **datatype** and **structuring** the
    output of an API.
    """

    web_socket_address: str


class GRPCResponseModel(BaseModel):
    """
    `GRPCResponseModel` is a validation class used for checking the **datatype** and **structuring** the output
    of an API.
    """

    gRPC_stream_address: str


class StreamingOptionModel(BaseModel):
    """
    `StreamingOptionModel` is a validation class that uses the above Enums for validating the **datatype**
    and **structure** of the data received from the user in an API call. This Class is used for providing the
    user to choose between the **Web-Socket** and **gRPC** for **streaming** data.
    """

    option: Options


class SocketPackageModel(BaseModel):
    """
    `SocketPackageModel` is a validation class for checking the **datatype** and structure of the data received from
    the user in an API call. This class is used for the checking the data packet received on the Web-Socker.
    """

    AccV: float
    AccML: float
    AccAP: float
