import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from keras.models import Model
from tensorflow.keras import layers as tkl


class ConstructRNN:
    """
    ConstructRNN class is used for building Rnn based model.
    """

    def __init__(self, drop_rate: float = 0.2, fun_type: str = 'tanh'):
        self.do_rate = drop_rate
        self.activation_type = fun_type

    def _lstm_layers(self, ly_in: KerasTensor, nodes: int):
        """
        LSTM layers method is used to construct a 6 layered LSTM network.
        :param ly_in: ly_in takes the input layer.
        :param nodes: nodes value is used for mentioning the number of neurons in each layer.
        """
        ll_in = tkl.LSTM(units=nodes, return_sequences=True, activation=self.activation_type, name='LSTM-1')(ly_in)
        dp_l1 = tkl.Dropout(self.do_rate, name='Drop-LSTM-1')(ll_in)
        ll_2 = tkl.LSTM(units=nodes, return_sequences=True, activation=self.activation_type, name='LSTM-2')(dp_l1)
        dp_l2 = tkl.Dropout(self.do_rate, name='Drop-LSTM-2')(ll_2)
        ll_3 = tkl.LSTM(units=nodes, return_sequences=True, activation=self.activation_type, name='LSTM-3')(dp_l2)
        dp_l3 = tkl.Dropout(self.do_rate, name='Drop-LSTM-3')(ll_3)
        ll_4 = tkl.LSTM(units=nodes, return_sequences=True, activation=self.activation_type, name='LSTM-4')(dp_l3)
        dp_l4 = tkl.Dropout(self.do_rate, name='Drop-LSTM-4')(ll_4)
        ll_5 = tkl.LSTM(units=nodes, return_sequences=True, activation=self.activation_type, name='LSTM-5')(dp_l4)
        dp_l5 = tkl.Dropout(self.do_rate, name='Drop-LSTM-5')(ll_5)
        ll_6 = tkl.LSTM(units=nodes, activation=self.activation_type, name='LSTM-6')(dp_l5)
        self._dp_out = tkl.Dropout(self.do_rate, name='Drop-LSTM-6')(ll_6)

    def build_lstm_model(self, data: tf.Tensor, size: tuple, nodes: int) -> Model:
        """
        Build lstm model method is used to build the full LSTM model.
        :param data: data is used to find it's mean and variance to normalize it.
        :param size: size is used to mention the input shape of the data.
        :param nodes: nodes value is used for mentioning the number of neurons in each layer.
        :return: Finally, this method returns the constructed model.
        """
        normalizer = tkl.Normalization()
        normalizer.adapt(data.map(lambda x, y: x))
        in_ly = tf.keras.Input(shape=size)
        norm_out = normalizer(in_ly)
        self._lstm_layers(norm_out, nodes)
        out_ly = tkl.Dense(units=3, activation='softmax', name='LSTM-out')(self._dp_out)
        return tf.keras.Model(inputs=in_ly, outputs=out_ly)
