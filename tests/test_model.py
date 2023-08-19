from src import Inference


class ModelMetrics(Inference):
    """
    ModelMetrics class inherits the Inference class to allow for testing the model.
    """

    def test_tdcsfog_rnn_model(self):
        """
        test_tdcsfog_rnn_model method is used to test the TDCSFOG RNN model.
        """
        self.load_tdcsfog_rnn_model()
        metrics = self.MODEL.evaluate(self.TEST_DATA)
        return metrics

    def test_tdcsfog_cnn_model(self):
        """
        test_tdcsfog_rnn_model method is used to test the TDCSFOG CNN model.
        """
        self.load_tdcsfog_cnn_model()
        metrics = self.MODEL.evaluate(self.TEST_DATA)
        return metrics

    def test_defog_rnn_model(self):
        """
        test_tdcsfog_rnn_model method is used to test the DEFOG RNN model.
        """
        self.load_defog_rnn_model()
        metrics = self.MODEL.evaluate(self.TEST_DATA)
        return metrics

    def test_defog_cnn_model(self):
        """
        test_tdcsfog_rnn_model method is used to test the DEFOG CNN model.
        """
        self.load_defog_cnn_model()
        metrics = self.MODEL.evaluate(self.TEST_DATA)
        return metrics
