Module api.app.controller.inference_controller
==============================================

Classes
-------

`InferenceController()`
:   `InferenceController` is a controller class used to load the particular model and use it to perform **prediction**.

    ### Ancestors (in MRO)

    * src.config.Inference
    * src.config.Modeling

    ### Methods

    `load_defog_cnn(self)`
    :   `load_defog_cnn` method **loads** defog cnn model into memory.

    `load_defog_rnn(self)`
    :   `load_defog_rnn` method **loads** defog rnn model into memory.

    `load_tdcsfog_cnn(self)`
    :   `load_tdcsfog_cnn` method **loads** tdcsfog cnn model into memory.

    `load_tdcsfog_rnn(self)`
    :   `load_tdcsfog_rnn` method **loads** tdcsfog rnn model into memory.

    `predict(self, data)`
    :   `predict` method is used to **predict** the freezing of gait from the given data.