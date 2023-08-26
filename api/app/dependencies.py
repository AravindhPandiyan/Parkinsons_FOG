from enum import Enum

from pydantic import BaseModel


class ArchitectureTypes(str, Enum):
    """
    ArchitectureTypes is an Enum class that limits the choices of Model Architectures that the user can select during
    API request.
    """

    RNN = "RNN"
    CNN = "CNN"


class UsableData(str, Enum):
    """
    UsableData is an Enum class that limits the choices of data that the user can select during API request.
    """

    TDCSFOG = "TDCSFOG"
    DEFOG = "DEFOG"


class APIResponseModel(BaseModel):
    """
    APIResponseModel is a validation class used for checking the datatype and structuring the output of an API.
    """

    details: str


class ModelTypesModel(BaseModel):
    """
    ModelTypesModel is a validation class that uses the above Enums for validating the datatype and structure of the
    data received from the user in an API call. This Class is used for limiting the user to choose the type of model and
    data used for training process.
    """

    use_data: UsableData
    architecture: ArchitectureTypes
