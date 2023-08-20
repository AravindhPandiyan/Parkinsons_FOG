from src import Preprocessing


class PreprocessorController:
    def __init__(self):
        self.processor = Preprocessing()

    def process_tdcsfog_rnn(self):
        self.processor.tdcsfog_rnn_model_preprocessing()

    def process_tdcsfog_cnn(self):
        self.processor.tdcsfog_cnn_model_preprocessing()

    def process_defog_rnn(self):
        self.processor.defog_rnn_model_preprocessing()

    def process_defog_cnn(self):
        self.processor.defog_cnn_model_preprocessing()
