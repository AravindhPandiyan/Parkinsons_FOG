Module api.app.router.preprocessing_router
==========================================

Functions
---------

    
`process(digest: api.app.dependencies.ModelTypesModel)`
:   `process` is an API route for **preprocessing** the different options of data in for each model architectures.
    
    Params:
        `digest`: digest is the data received from the user, containing the type of
        streaming requested.
    
    Returns:
        The return values of the function is dependent on the state of API.