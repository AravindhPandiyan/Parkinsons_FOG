import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from keras.models import Model
from tensorflow.keras import layers as tkl


class ConstructRNN:
    """
    ConstructRNN class is used for building Rnn based model.
    """

    def __init__(self, drop_rate: float = 0.2, fun_type: str = 'tanh'):
        self.dp_rate = drop_rate
        self.act_type = fun_type

    def _lstm_layers(self, ly_in: KerasTensor, nodes: int):
        """
        LSTM layers method is used to construct a 6 layered LSTM network.
        :param ly_in: ly_in takes the input layer.
        :param nodes: nodes value is used for mentioning the number of neurons in each layer.
        """
        ll_in = tkl.LSTM(units=nodes, return_sequences=True, activation=self.act_type, name='LSTM-1')(ly_in)
        dp_l1 = tkl.Dropout(self.dp_rate, name='Drop-LSTM-1')(ll_in)
        ll_2 = tkl.LSTM(units=nodes, return_sequences=True, activation=self.act_type, name='LSTM-2')(dp_l1)
        dp_l2 = tkl.Dropout(self.dp_rate, name='Drop-LSTM-2')(ll_2)
        ll_3 = tkl.LSTM(units=nodes, return_sequences=True, activation=self.act_type, name='LSTM-3')(dp_l2)
        dp_l3 = tkl.Dropout(self.dp_rate, name='Drop-LSTM-3')(ll_3)
        ll_4 = tkl.LSTM(units=nodes, return_sequences=True, activation=self.act_type, name='LSTM-4')(dp_l3)
        dp_l4 = tkl.Dropout(self.dp_rate, name='Drop-LSTM-4')(ll_4)
        ll_5 = tkl.LSTM(units=nodes, return_sequences=True, activation=self.act_type, name='LSTM-5')(dp_l4)
        dp_l5 = tkl.Dropout(self.dp_rate, name='Drop-LSTM-5')(ll_5)
        ll_6 = tkl.LSTM(units=nodes, activation=self.act_type, name='LSTM-6')(dp_l5)
        dp_out = tkl.Dropout(self.dp_rate, name='Drop-LSTM-6')(ll_6)
        return dp_out

    def build_model(self, data: tf.Tensor, size: tuple, nodes: int) -> Model:
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
        bl_out = self._lstm_layers(norm_out, nodes)
        out_ly = tkl.Dense(units=3, activation='softmax', name='LSTM-out')(bl_out)
        return tf.keras.Model(inputs=in_ly, outputs=out_ly)


class ConstructCNN:
    """
    ConstructRNN class is used for building Rnn based model.
    """

    def __init__(self, drop_rate: float = 0.2, fun_type: str = 'tanh'):
        self.dp_rate = drop_rate
        self.act_type = fun_type

    def _convo_layers(self, ly_in):
        cl_in = tkl.Conv1D(filters=128, kernel_size=3, strides=1, activation=self.act_type, name='Convo1D-in')(ly_in)
        ap1 = tkl.AveragePooling1D(pool_size=3, strides=1, name='AvgPooling1D-1')(cl_in)
        cl2 = tkl.Conv1D(filters=64, kernel_size=2, strides=1, activation=self.act_type, name='Convo1D-2')(ap1)
        ap2 = tkl.AveragePooling1D(pool_size=2, strides=1, name='AvgPooling1D-2')(cl2)
        cl3 = tkl.Conv1D(filters=32, kernel_size=2, strides=1, activation=self.act_type, name='Convo1D-3')(ap2)
        ap3 = tkl.AveragePooling1D(pool_size=2, strides=1, name='AvgPooling1D-3')(cl3)
        cl4 = tkl.Conv1D(filters=16, kernel_size=3, strides=1, activation=self.act_type, name='Convo1D-4')(ap3)
        ap4 = tkl.AveragePooling1D(pool_size=3, strides=1, name='AvgPooling1D-4')(cl4)
        cl5 = tkl.Conv1D(filters=8, kernel_size=3, strides=1, activation=self.act_type, name='Convo1D-5')(ap4)
        ap_out = tkl.AveragePooling1D(pool_size=3, strides=1, name='AvgPooling1D-out')(cl5)
        return ap_out

    def _dense_layers(self, ly_in, nodes: list):
        dl_in = tkl.Dense(units=nodes[0], activation=self.act_type, name='Ann-in')(ly_in)
        dp_1 = tkl.Dropout(self.dp_rate, name='Drop-Ann-1')(dl_in)
        dl_2 = tkl.Dense(units=nodes[1], activation=self.act_type, name='Ann-2')(dp_1)
        dp_2 = tkl.Dropout(self.dp_rate, name='Drop-Ann-2')(dl_2)
        dl_3 = tkl.Dense(units=nodes[2], activation=self.act_type, name='Ann-3')(dp_2)
        dp_3 = tkl.Dropout(self.dp_rate, name='Drop-Ann-3')(dl_3)
        dl_4 = tkl.Dense(units=nodes[3], activation=self.act_type, name='Ann-4')(dp_3)
        dp_4 = tkl.Dropout(self.dp_rate, name='Drop-Ann-4')(dl_4)
        dl_5 = tkl.Dense(units=nodes[4], activation=self.act_type, name='Ann-5')(dp_4)
        dp_5 = tkl.Dropout(self.dp_rate, name='Drop-Ann-5')(dl_5)
        dl_6 = tkl.Dense(units=nodes[5], activation=self.act_type, name='Ann-6')(dp_5)
        dp_out = tkl.Dropout(self.dp_rate, name='Drop-Ann-6')(dl_6)
        return dp_out

    def build_model(self, data: tf.Tensor, size: tuple, nodes: list) -> Model:
        normalizer = tkl.Normalization()
        normalizer.adapt(data.map(lambda x, y: x))
        in_ly = tf.keras.Input(shape=size)
        norm_out = normalizer(in_ly)
        bl_1 = self._convo_layers(norm_out)
        flat = tkl.Flatten()(bl_1)
        bl_2 = self._dense_layers(flat, nodes)
        out_ly = tkl.Dense(units=3, activation='softmax', name='Ann-out')(bl_2)
        return tf.keras.Model(inputs=in_ly, outputs=out_ly)
