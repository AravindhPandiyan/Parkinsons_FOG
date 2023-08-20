from src import Inference


class InferenceController:
    def __init__(self):
        self.infer = Inference()

    def load_tdcsfog_rnn(self):
        self.infer.load_tdcsfog_rnn_model()

    def load_tdcsfog_cnn(self):
        self.infer.load_tdcsfog_cnn_model()

    def load_defog_rnn(self):
        self.infer.load_defog_rnn_model()

    def load_defog_cnn(self):
        self.infer.load_defog_cnn_model()
