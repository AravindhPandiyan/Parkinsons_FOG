Module api.app.router.inference_router
======================================

Functions
---------

    
`load_model(load: api.app.dependencies.ModelTypesModel)`
:   `load_model` is an API route for **loading** different models into the system memory.

    Params:
        `load`: Data received from the user, containing a request for a specific model to be loaded into memory.
    
    Returns:
        The return values of the function are dependent on the current state of the API.

    
`streamers(choice: api.app.models.inference_model.StreamingOptionModel, request: starlette.requests.Request, background_tasks: starlette.background.BackgroundTasks)`
:   `streamers` is an API route for loading the choices between **Web Socket** and **gRPC** connections for
**inference streaming**.

    Params:
        `choice`: choice is the data received from the user, containing the type of
        **streaming** requested.
    
        `request`: request is the parameter passed to the funtion by the decorator funtion, It is used
        for getting the hosting url.
    
        `background_tasks`: background_tasks is the parameter passed to the funtion by the
        decorator funtion, It is used for running any long-running task or API-freezing task to run in the background.
    
    Returns:
        The return values of the function are dependent on the state of the API.

    
`web_socket_stream(websocket: starlette.websockets.WebSocket)`
:   `web_socket_stream` Upgrades the HTTP request response link to a **websocket** for enabling bidirectional
data transfer for prediction using the model.

    Params:
        `websocket`: The **websocket** parameter passed to the function by the decorator function. It
        is used to control the connection.