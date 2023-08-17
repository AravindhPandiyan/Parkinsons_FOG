from src import Modeling


class ModelingController:
    def __init__(self):
        self.modeler = Modeling()

    def build_tdcsfog(self):
        self.modeler.build_tdcsfog_rnn_model()

    def build_defog(self):
        self.modeler.build_tdcsfog_rnn_model()

    def train_tdcsfog(self):
        self.modeler.train_tdcsfog_model()

    def train_defog(self):
        self.modeler.train_defog_model()
