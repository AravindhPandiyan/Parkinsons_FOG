from tests import ModelMetrics


class MetricsController:

    def __init__(self):
        self.metrics = ModelMetrics()

    def test_tdcsfog_rnn(self):
        self.metrics.test_tdcsfog_rnn_model()

    def test_tdcsfog_cnn(self):
        self.metrics.test_tdcsfog_cnn_model()

    def test_defog_rnn(self):
        self.metrics.test_defog_rnn_model()

    def test_defog_cnn(self):
        self.metrics.test_defog_cnn_model()
