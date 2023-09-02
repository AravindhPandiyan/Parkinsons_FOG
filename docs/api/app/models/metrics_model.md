Module api.app.models.metrics_model
===================================

Classes
-------

`MetricsResponseModel(**data: Any)`
:   `MetricsResponseModel` is a validation class used for checking the datatype and structuring the output of an API.
    
    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises ValidationError if the input data cannot be parsed to form a valid model.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel
    * pydantic.utils.Representation

    ### Descendants

    * pydantic.main.MetricsResponseModel

    ### Class variables

    `auc: float`
    :

    `map: float`
    :