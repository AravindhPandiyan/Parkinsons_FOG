from src import Inference


class InferenceController(Inference):
    """
    `InferenceController` is a controller class used to load the particular model and use it to perform **prediction**.
    """

    def __init__(self):
        super().__init__()

    def load_tdcsfog_rnn(self):
        """
        `load_tdcsfog_rnn` method **loads** tdcsfog rnn model into memory.
        """
        return self.load_tdcsfog_rnn_model()

    def load_tdcsfog_cnn(self):
        """
        `load_tdcsfog_cnn` method **loads** tdcsfog cnn model into memory.
        """
        return self.load_tdcsfog_cnn_model()

    def load_defog_rnn(self):
        """
        `load_defog_rnn` method **loads** defog rnn model into memory.
        """
        return self.load_defog_rnn_model()

    def load_defog_cnn(self):
        """
        `load_defog_cnn` method **loads** defog cnn model into memory.
        """
        return self.load_defog_cnn_model()

    def predict(self, data):
        """
        `predict` method is used to **predict** the freezing of gait from the given data.
        """
        return self.predict_fog(data)
