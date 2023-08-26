from enum import Enum

from pydantic import BaseModel


class APIResponseModel(BaseModel):
    details: str


class UsableData(str, Enum):
    TDCSFOG = 'TDCSFOG'
    DEFOG = 'DEFOG'


class ArchitectureTypes(str, Enum):
    RNN = 'RNN'
    CNN = 'CNN'


class ModelTypesModel(BaseModel):
    use_data: UsableData
    architecture: ArchitectureTypes
