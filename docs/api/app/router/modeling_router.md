Module api.app.router.modeling_router
=====================================

Functions
---------

    
`build_model(build: api.app.dependencies.ModelTypesModel)`
:   `build_model` is an API route for **building** the different model's.

    Params:
        `build`: build is the data received from the user, containing the model requested by
        the user to be **constructed**.
    
    Returns:
        The return values of the function are dependent on the state of the API.

    
`train_model(train: api.app.dependencies.ModelTypesModel, background_tasks: starlette.background.BackgroundTasks)`
:   `train_model` is an API route for the **training** of the different model's.

    Params:
        `train`: train is the data received from the user, containing request of a specific
        model requested to be trained.
    
        `background_tasks`: background_tasks is the parameter passed to the funtion by the
        decorator funtion, It is used for running any long-running task or API-freezing task to run in the background.
    
    Returns:
        The return values of the function are dependent on the state of the API.