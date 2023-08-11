import hydra
import tensorflow as tf
from hydra import utils
from keras.models import Model
from omegaconf import DictConfig
from tensorflow.data.experimental import AUTOTUNE

from ._build_model import ConstructRNN
from ._load_data import _parse_tdcsfog_tfrecord, _parse_defog_tfrecord
from .preprocess import load_data, _tf_record_writer


@hydra.main(config_path='../config', config_name='Config.yaml', version_base='1.3')
def tdcsfog_preprocessing(config: DictConfig):
    """
    tdcsfog processing funtion is used to fetch and filter all the data that was tested in lab conditions.
    :param config: config parameter is used for accessing the configurations for the specific model.
    """
    current_path = utils.get_original_cwd() + '/'
    tdcsfog_paths = config.tdcsfog.preprocessing
    dataset = load_data(current_path + tdcsfog_paths.metadata, current_path + tdcsfog_paths.dataset)
    _tf_record_writer(dataset, current_path + tdcsfog_paths.tf_record_path, tdcsfog_paths.freq,
                      tdcsfog_paths.window_size, tdcsfog_paths.steps)


@hydra.main(config_path='../config', config_name='Config.yaml', version_base='1.3')
def defog_preprocessing(config: DictConfig):
    """
    defog processing funtion is used to fetch and filter all the data that was obtained from the subjects activities in their
    homes.
    :param config: config parameter is used for accessing the configurations for the specific model.
    """
    current_path = utils.get_original_cwd() + '/'
    defog_paths = config.defog.preprocessing
    dataset = load_data(current_path + defog_paths.metadata, current_path + defog_paths.dataset)
    dataset = dataset.loc[dataset.Valid.eq(True) & dataset.Task.eq(True)]
    dataset = dataset.drop(['Valid', 'Task'], axis=1).reset_index(drop=True)
    _tf_record_writer(dataset, current_path + defog_paths.tf_record_path, defog_paths.freq, defog_paths.window_size,
                      defog_paths.steps)


@hydra.main(config_path='../config', config_name='Config.yaml', version_base='1.3')
def build_tdcsfog_model(config: DictConfig) -> Model:
    """
    Build tdcsfog model funtion is used to create and return a rnn model for training on the tdcsfog data.
    :param config: config parameter is used for accessing the configurations for the specific model.
    :return: Finally, this function returns the model that was specifically constructed for tdcsfog data.
    """
    with tf.device("/GPU:0"):
        raw_dataset = tf.data.TFRecordDataset(config.tdcsfog.preprocessing.tf_record_path)
        dataset = raw_dataset.map(_parse_tdcsfog_tfrecord, num_parallel_calls=AUTOTUNE)
        precision = tf.keras.metrics.Precision()
        build = ConstructRNN()
        tdcsfog_model = build.build_lstm_model(dataset, eval(config.tdcsfog.input_size), config.tdcsfog.training_units)

    tdcsfog_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', precision])
    tdcsfog_model.summary()
    return tdcsfog_model


@hydra.main(config_path='../config', config_name='Config.yaml', version_base='1.3')
def build_defog_model(config: DictConfig) -> Model:
    """
    Build defog model funtion is used to create and return a rnn model for training on the defog data.
    :param config: config parameter is used for accessing the configurations for the specific model.
    :return: Finally, this function returns the model that was specifically constructed for defog data.
    """
    with tf.device("/GPU:0"):
        raw_dataset = tf.data.TFRecordDataset(config.defog.preprocessing.tf_record_path)
        dataset = raw_dataset.map(_parse_defog_tfrecord, num_parallel_calls=AUTOTUNE)
        precision = tf.keras.metrics.Precision()
        build = ConstructRNN()
        defog_model = build.build_lstm_model(dataset, eval(config.defog.input_size), config.defog.training_units)

    defog_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', precision])
    defog_model.summary()
    return defog_model
