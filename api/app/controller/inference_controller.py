from src import Inference


class InferenceController:
    def __init__(self):
        self.infer = Inference()

    def load_tdcsfog(self):
        self.infer.load_tdcsfog_model()

    def load_defog(self):
        self.infer.load_defog_model()
