from src import Preprocessing


class PreprocessorController(Preprocessing):
    """
    `PreprocessorController` is a controller class used for **processing** data.
    """

    def process_tdcsfog_rnn(self):
        """
        `process_tdcsfog_rnn` method is used for **processing** the tdcsfog data used to fit a rnn model.
        """
        self.tdcsfog_rnn_model_preprocessing()

    def process_tdcsfog_cnn(self):
        """
        `process_tdcsfog_cnn` method is used for **processing** the tdcsfog data used to fit a cnn model.
        """
        self.tdcsfog_cnn_model_preprocessing()

    def process_defog_rnn(self):
        """
        `process_defog_rnn` method is used for **processing** the defog data used to fit a rnn model.
        """
        self.defog_rnn_model_preprocessing()

    def process_defog_cnn(self):
        """
        `process_defog_cnn` method is used for **processing** the defog data used to fit a cnn model.
        """
        self.defog_cnn_model_preprocessing()
