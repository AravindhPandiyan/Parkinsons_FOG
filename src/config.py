from __future__ import annotations

import json

import hydra
import tensorflow as tf
from hydra import utils
from keras.models import Model
from omegaconf import DictConfig
from tensorflow.data.experimental import AUTOTUNE

from src.build_model import ConstructRNN, ConstructCNN
from src.load_data import TFRecordParsers
from src.preprocess import WindowWriter
from src.train_model import fitting

tf.keras.backend.set_floatx('float64')


class Preprocessing:
    """
    Preprocessing is a configured class which bring together all the other functions and class and makes you of them to
    process the data.
    """
    _CONFIG_PATH = '../config/process/'
    _VERSION = '1.3'
    _JSON_CONFIG = 'config/training.json'

    @classmethod
    def _len_writer(cls, type_d: str, data_len: int):
        """
        _len_writer method is used to write the length of the dataset into the json config file.
        :param type_d: type_d is used mention the type of data used.
        :param data_len: data_len is used for mentioning the data's length after processing.
        """
        with open(cls._JSON_CONFIG) as file:
            config = json.load(file)

        with open(cls._JSON_CONFIG, 'w') as file:
            config[f"{type_d}_len"] = data_len
            json.dump(config, file)

    @staticmethod
    @hydra.main(config_path=_CONFIG_PATH, config_name='tdcsfog_process.yaml', version_base=_VERSION)
    def tdcsfog_rnn_model_preprocessing(cfg: DictConfig):
        """
        tdcsfog_rnn_model_preprocessing method is used to fetch and filter all the data that was tested in lab
        conditions for rnn model.
        :param cfg: cfg parameter is used for accessing the configurations for the specific process types processing.
        """

        cfg_ps = cfg.power_spectrum
        current_path = utils.get_original_cwd() + '/'
        ww = WindowWriter(cfg.window_size, cfg.steps, cfg_ps.freq, cfg_ps.type)
        dataset = ww.load_csv_data(current_path + cfg.metadata, current_path + cfg.dataset)
        dataset_len = ww.tf_record_writer(dataset, current_path + cfg.tf_record_path.rnn)
        Preprocessing._len_writer(cfg.type, dataset_len)

    @staticmethod
    @hydra.main(config_path=_CONFIG_PATH, config_name='tdcsfog_process.yaml', version_base=_VERSION)
    def tdcsfog_cnn_model_preprocessing(cfg: DictConfig):
        """
        tdcsfog_cnn_model_preprocessing method is used to fetch and filter all the data that was tested in lab
        conditions for cnn model.
        :param cfg: cfg parameter is used for accessing the configurations for the specific process types processing.
        """

        current_path = utils.get_original_cwd() + '/'
        ww = WindowWriter(cfg.window_size, cfg.steps)
        dataset = ww.load_csv_data(current_path + cfg.metadata, current_path + cfg.dataset)
        dataset_len = ww.tf_record_writer(dataset, current_path + cfg.tf_record_path.cnn)
        Preprocessing._len_writer(cfg.type, dataset_len)

    @staticmethod
    @hydra.main(config_path=_CONFIG_PATH, config_name='defog_process.yaml', version_base=_VERSION)
    def defog_rnn_model_preprocessing(cfg: DictConfig):
        """
        defog_rnn_model_preprocessing method is used to fetch and filter all the data that was obtained from the
        subjects activities in their homes for rnn model.
        :param cfg: cfg parameter is used for accessing the configurations for the specific process types processing.
        """
        cfg_ps = cfg.power_spectrum
        current_path = utils.get_original_cwd() + '/'
        ww = WindowWriter(cfg.window_size, cfg.steps, cfg_ps.freq, cfg_ps.type)
        dataset = ww.load_csv_data(current_path + cfg.metadata, current_path + cfg.dataset)
        dataset = dataset.loc[dataset.Valid.eq(True) & dataset.Task.eq(True)]
        dataset = dataset.drop(['Valid', 'Task'], axis=1).reset_index(drop=True)
        dataset[['AccV', 'AccML', 'AccAP']] = dataset[['AccV', 'AccML', 'AccAP']] * 9.80665
        dataset_len = ww.tf_record_writer(dataset, current_path + cfg.tf_record_path.rnn)
        Preprocessing._len_writer(cfg.type, dataset_len)

    @staticmethod
    @hydra.main(config_path=_CONFIG_PATH, config_name='defog_process.yaml', version_base=_VERSION)
    def defog_cnn_model_preprocessing(cfg: DictConfig):
        """
        defog_rnn_model_preprocessing method is used to fetch and filter all the data that was obtained from the
        subjects activities in their homes for cnn model.
        :param cfg: cfg parameter is used for accessing the configurations for the specific process types processing.
        """
        current_path = utils.get_original_cwd() + '/'
        ww = WindowWriter(cfg.window_size, cfg.steps)
        dataset = ww.load_csv_data(current_path + cfg.metadata, current_path + cfg.dataset)
        dataset = dataset.loc[dataset.Valid.eq(True) & dataset.Task.eq(True)]
        dataset = dataset.drop(['Valid', 'Task'], axis=1).reset_index(drop=True)
        dataset[['AccV', 'AccML', 'AccAP']] = dataset[['AccV', 'AccML', 'AccAP']] * 9.80665
        dataset_len = ww.tf_record_writer(dataset, current_path + cfg.tf_record_path.cnn)
        Preprocessing._len_writer(cfg.type, dataset_len)


class Modeling:
    DEFOG_TRAIN_DATA = None
    DEFOG_VAL_DATA = None
    DEFOG_MODEL = None
    TDCSFOG_TRAIN_DATA = None
    TDCSFOG_VAL_DATA = None
    TDCSFOG_MODEL = None
    _CONFIG_PATH = '../config/model/'
    _VERSION = '1.3'
    _JSON_CONFIG = 'config/training.json'

    @classmethod
    def _build_model(cls, cfg: DictConfig, builder, type_d: str):
        with tf.device("/GPU:0"):
            raw_dataset = tf.data.TFRecordDataset(cfg.tf_record_path)
            raw_dataset = raw_dataset.shuffle(buffer_size=10000)

            with open(cls._JSON_CONFIG) as file:
                json_config = json.load(file)
                dataset_len = json_config[f'{type_d}_len']

            cls.TDCSFOG_TRAIN_DATA = raw_dataset.take(int(0.8 * dataset_len))
            cls.TDCSFOG_VAL_DATA = raw_dataset.skip(int(0.8 * dataset_len))
            _input_shape = eval(cfg.input_size)
            parser = TFRecordParsers(_input_shape[0], _input_shape[1])
            cls.TDCSFOG_TRAIN_DATA = cls.TDCSFOG_TRAIN_DATA.map(parser.tfrecord_parser, num_parallel_calls=AUTOTUNE)
            cls.TDCSFOG_VAL_DATA = cls.TDCSFOG_VAL_DATA.map(parser.tfrecord_parser, num_parallel_calls=AUTOTUNE)
            precision = tf.keras.metrics.Precision(name=f'{type_d}_precision')
            cls.TDCSFOG_MODEL = builder.build_model(cls.TDCSFOG_TRAIN_DATA, _input_shape[0], cfg.training_units)

        cls.TDCSFOG_MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', precision])
        cls.TDCSFOG_MODEL.summary()

    @staticmethod
    @hydra.main(config_path=_CONFIG_PATH, config_name='rnn_model.yaml', version_base=_VERSION)
    def build_tdcsfog_rnn_model(cfg: DictConfig):
        """
        build_tdcsfog_rnn_model method is used to create and return a rnn model for training on the tdcsfog data.
        :param cfg: cfg parameter is used for accessing the configurations for the specific model.
        """
        cfg = cfg.tdcsfog
        builder = ConstructRNN()
        Modeling._build_model(cfg, builder, 'tdcsfog')

    @staticmethod
    @hydra.main(config_path=_CONFIG_PATH, config_name='rnn_model.yaml', version_base=_VERSION)
    def build_defog_rnn_model(cfg: DictConfig):
        """
        Build defog model method is used to create and return a rnn model for training on the defog data.
        :param cfg: cfg parameter is used for accessing the configurations for the specific model.
        """
        cfg = cfg.defog
        builder = ConstructRNN()
        Modeling._build_model(cfg, builder, 'defog')

    @staticmethod
    @hydra.main(config_path=_CONFIG_PATH, config_name='cnn_model.yaml', version_base=_VERSION)
    def build_tdcsfog_cnn_model(cfg: DictConfig):
        """
        build_tdcsfog_rnn_model method is used to create and return a rnn model for training on the tdcsfog data.
        :param cfg: cfg parameter is used for accessing the configurations for the specific model.
        """
        cfg = cfg.tdcsfog
        builder = ConstructCNN()
        Modeling._build_model(cfg, builder, 'tdcsfog')

    @staticmethod
    @hydra.main(config_path=_CONFIG_PATH, config_name='cnn_model.yaml', version_base=_VERSION)
    def build_defog_cnn_model(cfg: DictConfig):
        """
        Build defog model method is used to create and return a rnn model for training on the defog data.
        :param cfg: cfg parameter is used for accessing the configurations for the specific model.
        """
        cfg = cfg.defog
        builder = ConstructCNN()
        Modeling._build_model(cfg, builder, 'defog')

    @classmethod
    def train_tdcsfog_model(cls) -> Model | None:
        """
        Train tdcsfog Model method is used to train the TDCSFOG model.
        :return: It if the necessary components are present it trains the model and return's it, if not it will return
        nothing.
        """
        if cls.TDCSFOG_TRAIN_DATA and cls.TDCSFOG_VAL_DATA and cls.TDCSFOG_MODEL:
            cls.TDCSFOG_MODEL = fitting(cls.TDCSFOG_MODEL, cls.TDCSFOG_TRAIN_DATA, cls.TDCSFOG_VAL_DATA,
                                        'tdcsfog')
            return cls.TDCSFOG_MODEL

        else:
            print('\nPlease First Build the TDCSFOG model to train it.')

    @classmethod
    def train_defog_model(cls) -> Model | None:
        """
        Train defog Model method is used to train the DEFOG model.
        :return: It if the necessary components are present it trains the model and return's it, if not it will return
        nothing.
        """
        if cls.DEFOG_TRAIN_DATA and cls.DEFOG_VAL_DATA and cls.DEFOG_MODEL:
            cls.DEFOG_MODEL = fitting(cls.DEFOG_MODEL, cls.DEFOG_TRAIN_DATA, cls.DEFOG_VAL_DATA,
                                      'defog')
            return cls.DEFOG_MODEL

        else:
            print('\nPlease First Build the DEFOG model to train it.')


class Inference(Modeling):
    def load_tdcsfog_rnn_model(self):
        self.build_tdcsfog_rnn_model()
        self.TDCSFOG_MODEL = self.TDCSFOG_MODEL.load_weights('models/ModelCheckpoint/tdcsfog/RNN/')

    def load_tdcsfog_cnn_model(self):
        self.build_tdcsfog_rnn_model()
        self.TDCSFOG_MODEL = self.TDCSFOG_MODEL.load_weights('models/ModelCheckpoint/tdcsfog/CNN/')

    def load_defog_rnn_model(self):
        self.build_defog_rnn_model()
        self.DEFOG_MODEL = self.DEFOG_MODEL.load_weights('models/ModelCheckpoint/defog/RNN')

    def load_defog_cnn_model(self):
        self.build_defog_rnn_model()
        self.DEFOG_MODEL = self.DEFOG_MODEL.load_weights('models/ModelCheckpoint/defog/CNN')
