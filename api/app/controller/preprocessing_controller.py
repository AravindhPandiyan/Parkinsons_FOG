from src import Preprocessing


class PreprocessorController:
    def __init__(self):
        self.processor = Preprocessing()

    def process_tdcsfog(self):
        self.processor.tdcsfog_rnn_model_preprocessing()

    def process_defog(self):
        self.processor.defog_rnn_model_preprocessing()
