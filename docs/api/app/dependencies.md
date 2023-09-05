Module api.app.dependencies
===========================

Classes
-------

`APIResponseModel(**data: Any)`
:   `APIResponseModel` is a validation class used for checking the **datatype** and **structuring** the output
of an API.

    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `__init__` uses `__pydantic_self__` instead of the more common `self` for the first arg to
    allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Class variables

    `details: str`
    :

    `model_config`
    :

    `model_fields`
    :

`ArchitectureTypes(value, names=None, *, module=None, qualname=None, type=None, start=1)`
:   `ArchitectureTypes` is an **Enum** class that **limits** the choices of **Model Architectures** that the
user can select during API request.

    ### Ancestors (in MRO)

    * builtins.str
    * enum.Enum

    ### Class variables

    `CNN`
    :

    `RNN`
    :

`ModelTypesModel(**data: Any)`
:   `ModelTypesModel` is a validation class that uses the above **Enums** for validating the **datatype**
and **structure** of the data received from the user in an API call. This Class is used for limiting the
user to choose the type of model and data used for **training** process.

    Create a new model by parsing and validating input data from keyword arguments.
    
    Raises [`ValidationError`][pydantic_core.ValidationError] if the input data cannot be
    validated to form a valid model.
    
    `__init__` uses `__pydantic_self__` instead of the more common `self` for the first arg to
    allow `self` as a field name.

    ### Ancestors (in MRO)

    * pydantic.main.BaseModel

    ### Class variables

    `architecture: api.app.dependencies.ArchitectureTypes`
    :

    `model_config`
    :

    `model_fields`
    :

    `use_data: api.app.dependencies.UsableData`
    :

`UsableData(value, names=None, *, module=None, qualname=None, type=None, start=1)`
:   `UsableData` is an **Enum** class that **limits** the choices of data that the user can select during
API request.

    ### Ancestors (in MRO)

    * builtins.str
    * enum.Enum

    ### Class variables

    `DEFOG`
    :

    `TDCSFOG`
    :