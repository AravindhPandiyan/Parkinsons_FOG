import hydra
import tensorflow as tf
from hydra import utils
from keras.models import Model
from omegaconf import DictConfig
from tensorflow.data.experimental import AUTOTUNE

from ._build_model import ConstructRNN
from ._load_data import _parse_tdcsfog_tfrecord, _parse_defog_tfrecord
from .preprocess import load_data, _tf_record_writer
from .train_model import fitting


class Preprocessing:

    @staticmethod
    @hydra.main(config_path='../config', config_name='modeling.yaml', version_base='1.3')
    def tdcsfog_preprocessing(config: DictConfig):
        """
        tdcsfog processing funtion is used to fetch and filter all the data that was tested in lab conditions.
        """
        tdcsfog_paths = config.tdcsfog.preprocessing
        current_path = utils.get_original_cwd() + '/'
        dataset = load_data(current_path + tdcsfog_paths.metadata, current_path + tdcsfog_paths.dataset)
        _tf_record_writer(dataset, current_path + tdcsfog_paths.tf_record_path, tdcsfog_paths.freq,
                          tdcsfog_paths.window_size, tdcsfog_paths.steps)

    @staticmethod
    @hydra.main(config_path='../config', config_name='modeling.yaml', version_base='1.3')
    def defog_preprocessing(config: DictConfig):
        """
        defog processing funtion is used to fetch and filter all the data that was obtained from the subjects activities in their
        homes.
        """
        defog_paths = config.defog.preprocessing
        current_path = utils.get_original_cwd() + '/'
        dataset = load_data(current_path + defog_paths.metadata, current_path + defog_paths.dataset)
        dataset = dataset.loc[dataset.Valid.eq(True) & dataset.Task.eq(True)]
        dataset = dataset.drop(['Valid', 'Task'], axis=1).reset_index(drop=True)
        dataset[['AccV', 'AccML', 'AccAP']] = dataset[['AccV', 'AccML', 'AccAP']] * 9.80665
        _tf_record_writer(dataset, current_path + defog_paths.tf_record_path, defog_paths.freq, defog_paths.window_size,
                          defog_paths.steps)


class Modeling:
    DEFOG_TRAIN_DATA = None
    DEFOG_VAL_DATA = None
    DEFOG_MODEL = None
    TDCSFOG_TRAIN_DATA = None
    TDCSFOG_VAL_DATA = None
    TDCSFOG_MODEL = None

    @staticmethod
    @hydra.main(config_path='../config', config_name='modeling.yaml', version_base='1.3')
    def build_tdcsfog_model(config: DictConfig):
        """
        Build tdcsfog model method is used to create and return a rnn model for training on the tdcsfog data.
        :param config: config parameter is used for accessing the configurations for the specific model.
        """
        with tf.device("/GPU:0"):
            raw_dataset = tf.data.TFRecordDataset(config.tdcsfog.preprocessing.tf_record_path)


            Modeling.TDCSFOG_DATASET = raw_dataset.map(_parse_tdcsfog_tfrecord, num_parallel_calls=AUTOTUNE)
            precision = tf.keras.metrics.Precision()
            build = ConstructRNN()
            Modeling.TDCSFOG_MODEL = build.build_lstm_model(Modeling.TDCSFOG_DATASET, eval(config.tdcsfog.input_size),
                                                            config.tdcsfog.training_units)

        Modeling.TDCSFOG_MODEL.compile(optimizer='adam', loss='categorical_crossentropy',
                                       metrics=['accuracy', precision])
        Modeling.TDCSFOG_MODEL.summary()

    @staticmethod
    @hydra.main(config_path='../config', config_name='modeling.yaml', version_base='1.3')
    def build_defog_model(config: DictConfig):
        """
        Build defog model method is used to create and return a rnn model for training on the defog data.
        :param config: config parameter is used for accessing the configurations for the specific model.
        """
        with tf.device("/GPU:0"):
            raw_dataset = tf.data.TFRecordDataset(config.defog.preprocessing.tf_record_path)
            Modeling.DEFOG_DATASET = raw_dataset.map(_parse_defog_tfrecord, num_parallel_calls=AUTOTUNE)
            precision = tf.keras.metrics.Precision()
            build = ConstructRNN()
            Modeling.DEFOG_MODEL = build.build_lstm_model(Modeling.DEFOG_DATASET, eval(config.defog.input_size),
                                                          config.defog.training_units)

        Modeling.DEFOG_MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', precision])
        Modeling.DEFOG_MODEL.summary()

    def train_tdcsfog_model(self) -> Model:
        if self.TDCSFOG_DATASET and self.TDCSFOG_MODEL:
            self.TDCSFOG_MODEL = fitting(self.TDCSFOG_MODEL, self.TDCSFOG_DATASET)
            return self.TDCSFOG_MODEL

        else:
            print('\nPlease First Build the TDCSFOG model to train it.')

    def train_defog_model(self) -> Model:
        if self.DEFOG_DATASET and self.DEFOG_MODEL:
            self.DEFOG_MODEL = fitting(self.DEFOG_MODEL, self.DEFOG_DATASET)
            return self.DEFOG_MODEL

        else:
            print('\nPlease First Build the DEFOG model to train it.')
