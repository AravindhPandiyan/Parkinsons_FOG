import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor
from keras.models import Model
from omegaconf import DictConfig
from tensorflow.keras import layers as tkl

from logger_config import logger as log


class ConstructRNN:
    """
    `ConstructRNN` is **constructs RNM model**.
    """

    def __init__(self, drop_rate: float = 0.2, fun_type: str = "tanh"):
        """
        `ConstructRNN` class is used for building **RNN** based model.

        Params:
            `drop_rate`: drop_rate is used to mention the **rate of dropout layer's**.

            `fun_type`: fun_type is the type of **activation function**.
        """

        self.dp_rate = drop_rate
        self.act_type = fun_type

    def _lstm_layers(self, ly_in: KerasTensor, nodes: int) -> KerasTensor:
        """
        `_lstm_layers` is a private method used to construct a `6` layered LSTM network with dropouts.

        Params:
            `ly_in`: ly_in is input for the first layer of the block.

            `nodes`: nodes value is used for mentioning the number of **neurons** in each layer.

        Returns:
            Finally, this method returns the output of the last layer of the block.
        """

        ll_in = tkl.LSTM(
            units=nodes, return_sequences=True, activation=self.act_type, name="LSTM-1"
        )(ly_in)
        dp_l1 = tkl.Dropout(self.dp_rate, name="Drop-LSTM-1")(ll_in)
        ll_2 = tkl.LSTM(
            units=nodes, return_sequences=True, activation=self.act_type, name="LSTM-2"
        )(dp_l1)
        dp_l2 = tkl.Dropout(self.dp_rate, name="Drop-LSTM-2")(ll_2)
        ll_3 = tkl.LSTM(
            units=nodes, return_sequences=True, activation=self.act_type, name="LSTM-3"
        )(dp_l2)
        dp_l3 = tkl.Dropout(self.dp_rate, name="Drop-LSTM-3")(ll_3)
        ll_4 = tkl.LSTM(
            units=nodes, return_sequences=True, activation=self.act_type, name="LSTM-4"
        )(dp_l3)
        dp_l4 = tkl.Dropout(self.dp_rate, name="Drop-LSTM-4")(ll_4)
        ll_5 = tkl.LSTM(
            units=nodes, return_sequences=True, activation=self.act_type, name="LSTM-5"
        )(dp_l4)
        dp_l5 = tkl.Dropout(self.dp_rate, name="Drop-LSTM-5")(ll_5)
        ll_6 = tkl.LSTM(units=nodes, activation=self.act_type, name="LSTM-6")(dp_l5)
        dp_out = tkl.Dropout(self.dp_rate, name="Drop-LSTM-6")(ll_6)
        return dp_out

    def build_model(self, data: tf.Tensor, size: tuple, nodes: int) -> Model:
        """
        `build_model` method is uses the properties of the class to **build** the model.

        Params:
            `data`: data is used to find it's **mean and variance** to **normalize** it.

            `size`: size is used to mention the **input shape** of the data.

            `nodes`: nodes value is used for mentioning the number of **neurons** in each layer.

        Returns:
            Finally, this method returns the constructed model.
        """

        normalizer = tkl.Normalization()
        normalizer.adapt(data.map(lambda x, y: x))
        in_ly = tf.keras.Input(shape=size)
        norm_out = normalizer(in_ly)
        bl_out = self._lstm_layers(norm_out, nodes)
        out_ly = tkl.Dense(units=3, activation="softmax", name="LSTM-out")(bl_out)
        return tf.keras.Model(inputs=in_ly, outputs=out_ly)


class ConstructCNN:
    """
    `ConstructCNN` is **constructs CNN model**.
    """

    def __init__(self, drop_rate: float = 0.2, fun_type: str = "relu"):
        """
        `ConstructCNN` class is used for building **CNN** a based model.

        Params:
            `drop_rate`: drop_rate is used to mention the **rate of dropout layer's**.

            `fun_type`: fun_type is the type of **activation function**.
        """

        self.dp_rate = drop_rate
        self.act_type = fun_type

    def _convo_layers(self, ly_in: KerasTensor, filters: list) -> KerasTensor:
        """
        `_convo_layers` is a private method used to **construct** a `5` layered **1DConvo** with **AvgPooling**.

        Params:
            `ly_in`: ly_in is input for the first layer of the block.

            `filters`: filters value is used for mentioning the number of filters in each layer.

        Returns:
            Finally, this method returns the output of the last layer of the block.
        """

        cl_in = tkl.Conv1D(
            filters=filters[0],
            kernel_size=3,
            strides=1,
            activation=self.act_type,
            name="Convo1D-in",
        )(ly_in)
        ap1 = tkl.AveragePooling1D(pool_size=3, strides=1, name="AvgPooling1D-1")(cl_in)
        cl2 = tkl.Conv1D(
            filters=filters[1],
            kernel_size=2,
            strides=1,
            activation=self.act_type,
            name="Convo1D-2",
        )(ap1)
        ap2 = tkl.AveragePooling1D(pool_size=2, strides=1, name="AvgPooling1D-2")(cl2)
        cl3 = tkl.Conv1D(
            filters=filters[2],
            kernel_size=2,
            strides=1,
            activation=self.act_type,
            name="Convo1D-3",
        )(ap2)
        ap3 = tkl.AveragePooling1D(pool_size=2, strides=1, name="AvgPooling1D-3")(cl3)
        cl4 = tkl.Conv1D(
            filters=filters[3],
            kernel_size=3,
            strides=1,
            activation=self.act_type,
            name="Convo1D-4",
        )(ap3)
        ap4 = tkl.AveragePooling1D(pool_size=3, strides=1, name="AvgPooling1D-4")(cl4)
        cl5 = tkl.Conv1D(
            filters=filters[4],
            kernel_size=3,
            strides=1,
            activation=self.act_type,
            name="Convo1D-5",
        )(ap4)
        ap_out = tkl.AveragePooling1D(pool_size=3, strides=1, name="AvgPooling1D-out")(
            cl5
        )
        return ap_out

    def _dense_layers(self, ly_in: KerasTensor, units: list) -> KerasTensor:
        """
        `_dense_layers` is a private method used to construct a `6` layered **ANN** with **Dropouts**.

        Params:
            `ly_in`: ly_in is input for the first layer of the block.

            `units`: units value is used for mentioning the number of **neurons** in each layer.

        Returns:
            Finally, this method returns the output of the last layer of the block.
        """

        dl_in = tkl.Dense(units=units[0], activation=self.act_type, name="Ann-in")(
            ly_in
        )
        dp_1 = tkl.Dropout(self.dp_rate, name="Drop-Ann-1")(dl_in)
        dl_2 = tkl.Dense(units=units[1], activation=self.act_type, name="Ann-2")(dp_1)
        dp_2 = tkl.Dropout(self.dp_rate, name="Drop-Ann-2")(dl_2)
        dl_3 = tkl.Dense(units=units[2], activation=self.act_type, name="Ann-3")(dp_2)
        dp_3 = tkl.Dropout(self.dp_rate, name="Drop-Ann-3")(dl_3)
        dl_4 = tkl.Dense(units=units[3], activation=self.act_type, name="Ann-4")(dp_3)
        dp_4 = tkl.Dropout(self.dp_rate, name="Drop-Ann-4")(dl_4)
        dl_5 = tkl.Dense(units=units[4], activation=self.act_type, name="Ann-5")(dp_4)
        dp_5 = tkl.Dropout(self.dp_rate, name="Drop-Ann-5")(dl_5)
        dl_6 = tkl.Dense(units=units[5], activation=self.act_type, name="Ann-6")(dp_5)
        dp_out = tkl.Dropout(self.dp_rate, name="Drop-Ann-6")(dl_6)
        return dp_out

    def build_model(self, data: tf.Tensor, size: tuple, nodes: DictConfig) -> Model:
        """
        `build_model` method is using the properties of the class to build the model.

        Params:
            `data`: data is used to find it's **mean and variance** to **normalize** it.

            `size`: size is used to mention the **input shape** of the data.

            `nodes`: nodes value is used for mentioning the number of **neurons** in each layer.

        Returns:
            Finally, this method returns the constructed model.
        """

        normalizer = tkl.Normalization()
        normalizer.adapt(data.map(lambda x, y: x))
        in_ly = tf.keras.Input(shape=size)
        norm_out = normalizer(in_ly)
        bl_1 = self._convo_layers(norm_out, nodes.filter)
        flat = tkl.Flatten()(bl_1)
        bl_2 = self._dense_layers(flat, nodes.units)
        out_ly = tkl.Dense(units=3, activation="softmax", name="Ann-out")(bl_2)
        return tf.keras.Model(inputs=in_ly, outputs=out_ly)
