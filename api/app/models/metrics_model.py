from pydantic import BaseModel


class MetricsResponseModel(BaseModel):
    """
    `MetricsResponseModel` is a validation class used for checking the datatype and structuring the output of an API.
    """

    map: float
    auc: float
