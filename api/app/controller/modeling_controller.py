from src import Modeling


class ModelingController(Modeling):
    """
    ModelingController is a controller class used for modeling.
    """

    def __init__(self):
        self.training_status = None

    def build_tdcsfog_rnn(self):
        """
        build_tdcsfog_rnn method is used for constructing a tdcsfog data based rnn model.
        """

        self.build_tdcsfog_rnn_model()

    def build_tdcsfog_cnn(self):
        """
        build_tdcsfog_cnn method is used for constructing a tdcsfog data based cnn model.
        """
        self.build_tdcsfog_cnn_model()

    def build_defog_rnn(self):
        """
        build_defog_rnn method is used for constructing a defog data based rnn model.
        """
        self.build_defog_rnn_model()

    def build_defog_cnn(self):
        """
        build_defog_rnn method is used for constructing a defog data based cnn model.
        """
        self.build_defog_cnn_model()

    def train_tdcsfog_rnn(self):
        """
        train_tdcsfog_rnn method is used for training a rnn model on tdcsfog.
        """
        return self.train_tdcsfog_rnn_model()

    def train_tdcsfog_cnn(self):
        """
        train_tdcsfog_cnn method is used for training a cnn model on tdcsfog.
        """
        return self.train_tdcsfog_cnn_model()

    def train_defog_rnn(self):
        """
        train_defog_rnn method is used for training a rnn model on defog.
        """
        return self.train_defog_rnn_model()

    def train_defog_cnn(self):
        """
        train_defog_cnn method is used for training a cnn model on defog.
        """
        return self.train_defog_cnn_model()
