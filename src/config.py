from __future__ import annotations

import json
import os

import hydra
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra import utils
from keras.models import Model
from omegaconf import DictConfig
from tensorflow.data.experimental import AUTOTUNE

from logger_config import logger as log
from src.build_model import ConstructCNN, ConstructRNN
from src.load_data import TFRecordParsers
from src.preprocess import WindowWriter
from src.train_model import fitting

with open("config/mlflow.json") as file:
    mlflow_cfg = json.load(file)
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_cfg["tracking_username"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_cfg["tracking_password"]

mlflow.tensorflow.autolog()
tf.keras.backend.set_floatx("float64")


class Preprocessing:
    """
    `Preprocessing` is a **configured** class that bring together all the other functions and class and makes use of
    them to **process the data**.
    """

    _CONFIG_PATH = "../config/process/"
    _VERSION = "1.3"
    _JSON_CONFIG = "config/training.json"

    @classmethod
    def _len_writer(cls, type_d: str, data_len: int):
        """
        `_len_writer` method is used to write the **length** of the dataset into the json **config file**.

        Params:
            `type_d`: type_d is used to mention the **type** of data used.

            `data_len`: data_len is used for mentioning the data's **length** after **processing**.
        """
        log.info("Method Call")
        with open(cls._JSON_CONFIG) as file:
            config = json.load(file)

        with open(cls._JSON_CONFIG, "w") as file:
            config[f"{type_d}_len"] = data_len
            json.dump(config, file)

    @staticmethod
    @hydra.main(
        config_path=_CONFIG_PATH,
        config_name="tdcsfog_process.yaml",
        version_base=_VERSION,
    )
    def tdcsfog_rnn_model_preprocessing(cfg: DictConfig):
        """
        `tdcsfog_rnn_model_preprocessing` method is used to **fetch** and **filter** all the data that was
        tested in **lab conditions** for **rnn** model.

        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific process
            types **processing**.
        """
        log.info("Method Call")
        cfg_ps = cfg.power_spectrum
        current_path = utils.get_original_cwd() + "/"
        ww = WindowWriter(cfg.window_size, cfg.steps, cfg_ps.freq, cfg_ps.type)
        dataset = ww.load_csv_data(
            current_path + cfg.metadata, current_path + cfg.dataset
        )
        dataset_len = ww.tf_record_writer(
            dataset, current_path + cfg.tf_record_path.rnn
        )
        Preprocessing._len_writer(cfg.type, dataset_len)

    @staticmethod
    @hydra.main(
        config_path=_CONFIG_PATH,
        config_name="tdcsfog_process.yaml",
        version_base=_VERSION,
    )
    def tdcsfog_cnn_model_preprocessing(cfg: DictConfig):
        """
        `tdcsfog_cnn_model_preprocessing` method is used to **fetch** and **filter** all the data that was
        tested in **lab conditions** for **cnn** a model.

        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific process
            types **processing**.
        """
        log.info("Method Call")
        current_path = utils.get_original_cwd() + "/"
        ww = WindowWriter(cfg.window_size, cfg.steps)
        dataset = ww.load_csv_data(
            current_path + cfg.metadata, current_path + cfg.dataset
        )
        dataset_len = ww.tf_record_writer(
            dataset, current_path + cfg.tf_record_path.cnn
        )
        Preprocessing._len_writer(cfg.type, dataset_len)

    @staticmethod
    @hydra.main(
        config_path=_CONFIG_PATH,
        config_name="defog_process.yaml",
        version_base=_VERSION,
    )
    def defog_rnn_model_preprocessing(cfg: DictConfig):
        """
        `defog_rnn_model_preprocessing` method is used to **fetch** and **filter** all the data that was obtained
        from the **subjects activities** in their **homes** for **rnn** a model.

        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific process
            types **processing**.
        """
        log.info("Method Call")
        cfg_ps = cfg.power_spectrum
        current_path = utils.get_original_cwd() + "/"
        ww = WindowWriter(cfg.window_size, cfg.steps, cfg_ps.freq, cfg_ps.type)
        dataset = ww.load_csv_data(
            current_path + cfg.metadata, current_path + cfg.dataset
        )
        dataset = dataset.loc[dataset.Valid.eq(True) & dataset.Task.eq(True)]
        dataset = dataset.drop(["Valid", "Task"], axis=1).reset_index(drop=True)
        dataset[["AccV", "AccML", "AccAP"]] = (
            dataset[["AccV", "AccML", "AccAP"]] * 9.80665
        )
        dataset_len = ww.tf_record_writer(
            dataset, current_path + cfg.tf_record_path.rnn
        )
        Preprocessing._len_writer(cfg.type, dataset_len)

    @staticmethod
    @hydra.main(
        config_path=_CONFIG_PATH,
        config_name="defog_process.yaml",
        version_base=_VERSION,
    )
    def defog_cnn_model_preprocessing(cfg: DictConfig):
        """
        `defog_rnn_model_preprocessing` method is used to **fetch** and **filter** all the data that was obtained
        from the **subjects activities** in their **homes** for **cnn** a model.

        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific process
            types **processing**.

        """
        log.info("Method Call")
        current_path = utils.get_original_cwd() + "/"
        ww = WindowWriter(cfg.window_size, cfg.steps)
        dataset = ww.load_csv_data(
            current_path + cfg.metadata, current_path + cfg.dataset
        )
        dataset = dataset.loc[dataset.Valid.eq(True) & dataset.Task.eq(True)]
        dataset = dataset.drop(["Valid", "Task"], axis=1).reset_index(drop=True)
        dataset[["AccV", "AccML", "AccAP"]] = (
            dataset[["AccV", "AccML", "AccAP"]] * 9.80665
        )
        dataset_len = ww.tf_record_writer(
            dataset, current_path + cfg.tf_record_path.cnn
        )
        Preprocessing._len_writer(cfg.type, dataset_len)


class Modeling:
    """
    `Modeling` is a **configured** class which bring together all the other **functions** and **class** and makes
    use of them to **build amd train the model**.
    """

    TRAIN_DATA = None
    VAL_DATA = None
    TEST_DATA = None
    MODEL = None
    MODEL_TYPE = None
    _CONFIG_PATH = "../config/model/"
    _VERSION = "1.3"
    _JSON_CONFIG = "config/training.json"

    @staticmethod
    def _custom_map_loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        `_custom_map_loss` is used for calculating the **mean Average Precision loss** for the model.

        Params:
            `y_true`: y_true is the **actual values**.

            `y_pred`: y_pred is the **predicted values**.

        Returns:
            Finally, the **loss** values are calculated and returned. The **loss** value will be in **negative**.
        """
        y_true_float = tf.cast(y_true, tf.float64)
        intersection = tf.reduce_sum(tf.minimum(y_true_float, y_pred))
        union = tf.reduce_sum(tf.maximum(y_true_float, y_pred))
        iou = intersection / union
        loss = -iou
        return loss

    @classmethod
    def _build_model(
        cls, cfg: DictConfig, builder: ConstructCNN | ConstructRNN, type_d: str
    ):
        """
        `_build_model` is a **private method** meant to be only used withing the class for **building models**.
        Params:
            `cfg`: cfg is used to access the model **configuration**.

            `builder`: builder is used to **construct the model**.

            `type_d`: type_d is used for identifying the **type of data**.
        """
        log.info("Method Call")

        try:
            with tf.device("/GPU:0"):
                raw_dataset = tf.data.TFRecordDataset(cfg.tf_record_path)

                with open(cls._JSON_CONFIG) as file:
                    json_config = json.load(file)
                    dataset_len = json_config[f"{type_d}_len"]

                raw_dataset = raw_dataset
                train_len = int(0.8 * dataset_len)
                val_len = int(0.1 * dataset_len)
                cls.TRAIN_DATA = raw_dataset.take(train_len)
                remaining_data = raw_dataset.skip(train_len)
                remaining_data = remaining_data.shuffle(
                    buffer_size=val_len * 2, seed=json_config["shuffle_seed"]
                )
                cls.VAL_DATA = remaining_data.take(val_len)
                cls.TEST_DATA = remaining_data.skip(val_len)
                input_shape = eval(cfg.input_size)
                parser = TFRecordParsers(input_shape[0], input_shape[1])
                cls.TRAIN_DATA = cls.TRAIN_DATA.map(
                    parser.tfrecord_parser, num_parallel_calls=AUTOTUNE
                )
                cls.TRAIN_DATA = cls.TRAIN_DATA.shuffle(
                    buffer_size=json_config["buffer_size"],
                    reshuffle_each_iteration=True,
                )
                cls.VAL_DATA = cls.VAL_DATA.map(
                    parser.tfrecord_parser, num_parallel_calls=AUTOTUNE
                )
                cls.TEST_DATA = cls.TEST_DATA.map(
                    parser.tfrecord_parser, num_parallel_calls=AUTOTUNE
                )
                cls.MODEL = builder.build_model(
                    cls.TRAIN_DATA, input_shape[0], cfg.training_units
                )
                auc = tf.keras.metrics.AUC(name="auc")

            cls.MODEL.compile(
                optimizer="adam", loss=cls._custom_map_loss, metrics=[auc]
            )
            cls.MODEL.summary()

        except tf.errors.NotFoundError as e:
            msg = "Data for training couldn't be found."
            log.error(msg + ": " + str(e))
            print(f"\n{msg}")

    @staticmethod
    @hydra.main(
        config_path=_CONFIG_PATH, config_name="rnn_model.yaml", version_base=_VERSION
    )
    def build_tdcsfog_rnn_model(cfg: DictConfig):
        """
        `build_tdcsfog_rnn_model` method is used to construct a **rnn model** for **training** on the **tdcsfog data**.

        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific model.
        """
        log.info("Method Call")
        cfg = cfg.tdcsfog
        builder = ConstructRNN()
        Modeling.MODEL_TYPE = "TDCSFOG_RNN"
        Modeling._build_model(cfg, builder, "tdcsfog")

    @staticmethod
    @hydra.main(
        config_path=_CONFIG_PATH, config_name="cnn_model.yaml", version_base=_VERSION
    )
    def build_tdcsfog_cnn_model(cfg: DictConfig):
        """
        `build_tdcsfog_cnn_model` method is used to construct a **cnn model** for **training** on the **tdcsfog data**.

        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific model.
        """
        log.info("Method Call")
        cfg = cfg.tdcsfog
        builder = ConstructCNN()
        Modeling.MODEL_TYPE = "TDCSFOG_CNN"
        Modeling._build_model(cfg, builder, "tdcsfog")

    @staticmethod
    @hydra.main(
        config_path=_CONFIG_PATH, config_name="rnn_model.yaml", version_base=_VERSION
    )
    def build_defog_rnn_model(cfg: DictConfig):
        """
        `build_defog_rnn_model` method is used to construct a **rnn model** for **training** on the **defog data**.

        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific model.
        """
        log.info("Method Call")
        cfg = cfg.defog
        builder = ConstructRNN()
        Modeling.MODEL_TYPE = "DEFOG_RNN"
        Modeling._build_model(cfg, builder, "defog")

    @staticmethod
    @hydra.main(
        config_path=_CONFIG_PATH, config_name="cnn_model.yaml", version_base=_VERSION
    )
    def build_defog_cnn_model(cfg: DictConfig):
        """
        `build_defog_cnn_model` method is used to construct a **cnn model** for **training** on the **defog data**.

        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific model.
        """
        log.info("Method Call")
        cfg = cfg.defog
        builder = ConstructCNN()
        Modeling.MODEL_TYPE = "DEFOG_CNN"
        Modeling._build_model(cfg, builder, "defog")

    @classmethod
    def train_tdcsfog_rnn_model(cls) -> Model | str:
        """
        `train_tdcsfog_rnn_model` method is used to train the **TDCSFOG RNN** model.

        Returns:
            It returns the **model** if the necessary **components** are present to **train** the **model** and
            return's a **warning string**, if the model is not **constructed** first.
        """
        log.info("Method Call")

        if (
            cls.TRAIN_DATA
            and cls.VAL_DATA
            and cls.MODEL
            and cls.MODEL_TYPE == "TDCSFOG_RNN"
        ):
            cls.MODEL = fitting(cls.MODEL, cls.TRAIN_DATA, cls.VAL_DATA, "tdcsfog/RNN/")
            return cls.MODEL

        else:
            msg = "Please First Build the TDCSFOG RNN model to train it."
            log.warning(msg)
            print(f"\n{msg}")
            return msg

    @classmethod
    def train_tdcsfog_cnn_model(cls) -> Model | str:
        """
        `train_tdcsfog_cnn_model` method is used to train the **TDCSFOG CNN** model.

        Returns:
            It returns the **model** if the necessary **components** are present to **train** the **model** and
            return's a **warning string**, if the model is not **constructed** first.
        """
        log.info("Method Call")

        if (
            cls.TRAIN_DATA
            and cls.VAL_DATA
            and cls.MODEL
            and cls.MODEL_TYPE == "TDCSFOG_CNN"
        ):
            cls.MODEL = fitting(cls.MODEL, cls.TRAIN_DATA, cls.VAL_DATA, "tdcsfog/CNN/")
            return cls.MODEL

        else:
            msg = "Please First Build the TDCSFOG CNN model to train it."
            log.warning(msg)
            print(f"\n{msg}")
            return msg

    @classmethod
    def train_defog_rnn_model(cls) -> Model | str:
        """
        `train_defog_rnn_model` method is used to train the **DEFOG RNN** model.

        Returns:
            It returns the **model** if the necessary **components** are present to **trains** the **model** and
            return's a **warning string**, if the model is not **constructed** first.
        """
        log.info("Method Call")

        if (
            cls.TRAIN_DATA
            and cls.VAL_DATA
            and cls.MODEL
            and cls.MODEL_TYPE == "DEFOG_RNN"
        ):
            cls.MODEL = fitting(cls.MODEL, cls.TRAIN_DATA, cls.VAL_DATA, "defog/RNN/")
            return cls.MODEL

        else:
            msg = "Please First Build the DEFOG RNN model to train it."
            log.warning(msg)
            print(f"\n{msg}")
            return msg

    @classmethod
    def train_defog_cnn_model(cls) -> Model | str:
        """
        `train_defog_cnn_model` method is used to **train** the **DEFOG CNN** model.

        Returns:
            It returns the **model** if the necessary **components** are present to **trains** the **model** and
            return's a **warning string**, if not it will return a **warning message**.
        """
        log.info("Method Call")
        if (
            cls.TRAIN_DATA
            and cls.VAL_DATA
            and cls.MODEL
            and cls.MODEL_TYPE == "DEFOG_CNN"
        ):
            cls.MODEL = fitting(cls.MODEL, cls.TRAIN_DATA, cls.VAL_DATA, "defog/CNN/")
            return cls.MODEL

        else:
            msg = "Please First Build the DEFOG CNN model to train it."
            log.warning(msg)
            print(f"\n{msg}")
            return msg


class Inference(Modeling):
    """
    `Inference` class **inherits** the **Modeling** class to allow for **loading of model, and prediction**.
    """

    def __init__(self):
        log.info("class Initialization")
        self.window_size = None
        self.steps = None
        self._m_type = None
        self._d_type = None
        self._processor = WindowWriter()

    def load_tdcsfog_rnn_model(self) -> str | None:
        """
        `load_tdcsfog_rnn_model` method is used to **load** the **TDCSFOG RNN** model.

        Returns:
            If there are **no checkpoints** of the **tdcsfog rnn model** it will return a **warning message**,
            else nothing.
        """
        log.info("Method Call")

        try:
            self._d_type = "tdcsfog"
            self._m_type = "RNN"
            self.build_tdcsfog_rnn_model()

            with open(self._JSON_CONFIG) as file:
                cfg = json.load(file)
                self.MODEL.load_weights(f"{cfg['checkpoint_loc']}tdcsfog/RNN/")

        except IsADirectoryError as w:
            msg = "Please First Train the TDCSFOG RNN model."
            log.warning(msg + ": " + str(w))
            print(f"\n{msg}")
            return msg

        with open("config/inference.json") as file:
            cfg = json.load(file)
            cfg = cfg[self._d_type]
            self.window_size = cfg["window_size"]
            self.steps = cfg["steps"]

    def load_tdcsfog_cnn_model(self) -> str | None:
        """
        `load_tdcsfog_rnn_model` method is used to **load** the **TDCSFOG CNN** model.

        Returns:
            If there are **no checkpoints** of the **tdcsfog cnn model** it will return a **warning message**,
            else nothing.
        """
        log.info("Method Call")

        try:
            self._d_type = "tdcsfog"
            self._m_type = "CNN"
            self.build_tdcsfog_cnn_model()

            with open(self._JSON_CONFIG) as file:
                cfg = json.load(file)
                self.MODEL.load_weights(f"{cfg['checkpoint_loc']}tdcsfog/RNN/")

        except IsADirectoryError as w:
            msg = "Please First Train the TDCSFOG CNN model."
            log.warning(msg + ": " + str(w))
            print(f"\n{msg}")
            return msg

        with open("config/inference.json") as file:
            cfg = json.load(file)
            cfg = cfg[self._d_type]
            self.window_size = cfg["window_size"]
            self.steps = cfg["steps"]

    def load_defog_rnn_model(self) -> str | None:
        """
        `load_tdcsfog_rnn_model` method is used to **load** the **DEFOG RNN** model.

        Returns:
            If there are **no checkpoints** of the **defog rnn model** it will return a **warning message**,
            else nothing.
        """
        log.info("Method Call")

        try:
            self._d_type = "defog"
            self._m_type = "RNN"
            self.build_defog_rnn_model()

            with open(self._JSON_CONFIG) as file:
                cfg = json.load(file)
                self.MODEL.load_weights(f"{cfg['checkpoint_loc']}tdcsfog/RNN/")

        except IsADirectoryError as w:
            msg = "Please First Train the DEFOG RNN model."
            log.warning(msg + ": " + str(w))
            print(f"\n{msg}")
            return msg

        with open("config/inference.json") as file:
            cfg = json.load(file)
            cfg = cfg[self._d_type]
            self.window_size = cfg["window_size"]
            self.steps = cfg["steps"]

    def load_defog_cnn_model(self) -> str | None:
        """
        `load_tdcsfog_rnn_model` method is used to load the **DEFOG CNN** model.

        Returns:
            If there are **no checkpoints** of the **defog cnn** model it will return a **warning message**,
            else nothing.
        """
        log.info("Method Call")

        try:
            self._d_type = "defog"
            self._m_type = "CNN"
            self.build_defog_cnn_model()

            with open(self._JSON_CONFIG) as file:
                cfg = json.load(file)
                self.MODEL.load_weights(f"{cfg['checkpoint_loc']}tdcsfog/RNN/")

        except IsADirectoryError as w:
            msg = "Please First Train the DEFOG CNN model."
            log.warning(msg + ": " + str(w))
            print(f"\n{msg}")
            return msg

        with open("config/inference.json") as file:
            cfg = json.load(file)
            cfg = cfg[self._d_type]
            self.window_size = cfg["window_size"]
            self.steps = cfg["steps"]

    def predict_fog(self, data: pd.DataFrame) -> list | str:
        """
        `predict_fog` method is used for **performing prediction** on the **ML model** that has been **loaded**
        into memory.

        Params:
            `data`: data is the series of data from **accelerometer** used for **predicting** if there is a
            **fog detected**.

        Returns:
            Finally, it returns the **prediction** list, or if the model is **not loaded** it will return a
            **warning string**.
        """
        log.info("Method Call")

        if self.window_size and self.steps:
            if self._m_type == "RNN":
                data, _ = self._processor.window_processing(data)
                data = data.reshape(shape=(1, self.window_size // 2 + 1, 3))

            else:
                data = np.array([data.values])

            res = self.MODEL.predict(data)
            return res

        else:
            msg = "Please First Load the model."
            log.warning(msg)
            print(f"\n{msg}")
            return msg
