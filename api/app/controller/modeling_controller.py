from src import Modeling


class ModelingController:
    def __init__(self):
        self.modeler = Modeling()

    def build_tdcsfog_rnn(self):
        self.modeler.build_tdcsfog_rnn_model()

    def build_tdcsfog_cnn(self):
        self.modeler.build_tdcsfog_cnn_model()

    def build_defog_rnn(self):
        self.modeler.build_defog_rnn_model()

    def build_defog_cnn(self):
        self.modeler.build_defog_cnn_model()

    def train_tdcsfog_rnn(self):
        self.modeler.train_tdcsfog_rnn_model()

    def train_tdcsfog_cnn(self):
        self.modeler.train_tdcsfog_cnn_model()

    def train_defog_rnn(self):
        self.modeler.train_defog_rnn_model()

    def train_defog_cnn(self):
        self.modeler.train_defog_cnn_model()
