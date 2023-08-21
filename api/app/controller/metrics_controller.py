from tests import ModelMetrics


class MetricsController:
    """
    MetricsController is a controller class used to test the particular model.
    """

    def __init__(self):
        self.metrics = ModelMetrics()

    def test_tdcsfog_rnn(self):
        """
        test_tdcsfog_rnn method helps to test the tdcsfog rnn model.
        """
        return self.metrics.test_tdcsfog_rnn_model()

    def test_tdcsfog_cnn(self):
        """
        test_tdcsfog_cnn method helps to test the tdcsfog cnn model.
        """
        return self.metrics.test_tdcsfog_cnn_model()

    def test_defog_rnn(self):
        """
        test_defog_rnn method helps to test the defog rnn model.
        """
        return self.metrics.test_defog_rnn_model()

    def test_defog_cnn(self):
        """
        test_defog_cnn method helps to test the defog cnn model.
        """
        return self.metrics.test_defog_cnn_model()
