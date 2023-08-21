from src import Preprocessing


class PreprocessorController:
    """
    PreprocessorController is a controller class used for processing data.
    """

    def __init__(self):
        self.processor = Preprocessing()

    def process_tdcsfog_rnn(self):
        """
        process_tdcsfog_rnn method is used for processing the tdcsfog data used to fit a rnn model.
        """
        self.processor.tdcsfog_rnn_model_preprocessing()

    def process_tdcsfog_cnn(self):
        """
        process_tdcsfog_cnn method is used for processing the tdcsfog data used to fit a cnn model.
        """
        self.processor.tdcsfog_cnn_model_preprocessing()

    def process_defog_rnn(self):
        """
        process_defog_rnn method is used for processing the defog data used to fit a rnn model.
        """
        self.processor.defog_rnn_model_preprocessing()

    def process_defog_cnn(self):
        """
        process_defog_cnn method is used for processing the defog data used to fit a cnn model.
        """
        self.processor.defog_cnn_model_preprocessing()
