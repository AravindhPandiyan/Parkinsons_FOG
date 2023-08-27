from __future__ import annotations

from src import Inference


class ModelMetrics(Inference):
    """
    `ModelMetrics` class **inherits** the **Inference** class to allow for **testing** the model.
    """

    def _test_model(self) -> dict:
        """
        `_test_model` is a private method used by the class to run the **evaluation** and return the **metrics** value.

        Returns:
            Finally, this method returns the calculated metrics in the form of a dictionary.
        """
        metrics = self.MODEL.evaluate(self.TEST_DATA.batch(1))
        metrics = dict(map=round(metrics[0] * -1, 4), auc=round(metrics[1], 4))
        print(f'Mean Average Precision(mAP): {metrics["map"]}')
        print(f'Area Under the Curve(AUC): {metrics["auc"]}')
        return metrics

    def test_tdcsfog_rnn_model(self) -> dict | str:
        """
        `test_tdcsfog_rnn_model` method is used to test the TDCSFOG RNN model.

        Returns:
            Finally, this method returns the calculated metrics for TDCSFOG RNN model in the form of a dictionary.
        """
        msg = self.load_tdcsfog_rnn_model()

        if msg:
            return msg
        else:
            return self._test_model()

    def test_tdcsfog_cnn_model(self) -> dict | str:
        """
        `test_tdcsfog_rnn_model` method is used to test the TDCSFOG CNN model.

        Returns:
            Finally, this method returns the calculated metrics for TDCSFOG CNN model in the form of a dictionary.
        """
        msg = self.load_tdcsfog_cnn_model()

        if msg:
            return msg
        else:
            return self._test_model()

    def test_defog_rnn_model(self) -> dict | str:
        """
        `test_tdcsfog_rnn_model` method is used to test the DEFOG RNN model.

        Returns:
            Finally, this method returns the calculated metrics for DEFOG RNN model in the form of a dictionary.
        """
        msg = self.load_defog_rnn_model()

        if msg:
            return msg
        else:
            return self._test_model()

    def test_defog_cnn_model(self) -> dict | str:
        """
        `test_tdcsfog_rnn_model` method is used to test the DEFOG CNN model.

        Returns:
            Finally, this method returns the calculated metrics for DEFOG CNN model in the form of a dictionary.
        """
        msg = self.load_defog_cnn_model()

        if msg:
            return msg
        else:
            return self._test_model()
