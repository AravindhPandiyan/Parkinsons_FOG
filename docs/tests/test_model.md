Module tests.test_model
=======================

Classes
-------

`ModelMetrics()`
:   `ModelMetrics` class **inherits** the **Inference** class to allow for **testing** the model.

    ### Ancestors (in MRO)

    * src.config.Inference
    * src.config.Modeling

    ### Descendants

    * api.app.controller.metrics_controller.MetricsController

    ### Methods

    `test_defog_cnn_model(self) ‑> dict | str`
    :   `test_tdcsfog_rnn_model` method is used to test the DEFOG CNN model.
        
        Returns:
            Finally, this method returns the calculated metrics for a DEFOG CNN model in the form of a dictionary.

    `test_defog_rnn_model(self) ‑> dict | str`
    :   `test_tdcsfog_rnn_model` method is used to test the DEFOG RNN model.
        
        Returns:
            Finally, this method returns the calculated metrics for a DEFOG RNN model in the form of a dictionary.

    `test_tdcsfog_cnn_model(self) ‑> dict | str`
    :   `test_tdcsfog_rnn_model` method is used to test the TDCSFOG CNN model.
        
        Returns:
            Finally, this method returns the calculated metrics for a TDCSFOG CNN model in the form of a dictionary.

    `test_tdcsfog_rnn_model(self) ‑> dict | str`
    :   `test_tdcsfog_rnn_model` method is used to test the TDCSFOG RNN model.
        
        Returns:
            Finally, this method returns the calculated metrics for a TDCSFOG RNN model in the form of a dictionary.