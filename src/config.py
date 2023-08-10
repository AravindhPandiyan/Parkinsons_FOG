import hydra
from hydra import utils
from omegaconf import DictConfig

from preprocess import load_data, tf_record_writer


@hydra.main(config_path='../config', config_name='Config.yaml', version_base='1.3')
def tdcsfog_main(config: DictConfig):
    """
    tdcsfog main funtion is used to fetch and filter all the data that was tested in lab conditions.
    :param config: config parameter is used for accessing the configurations for the specific model.
    """
    current_path = utils.get_original_cwd() + '/'
    tdcsfog_paths = config.tdcsfog.preprocessing
    dataset = load_data(current_path + tdcsfog_paths.metadata, current_path + tdcsfog_paths.dataset)
    tf_record_writer(dataset, current_path + tdcsfog_paths.tf_record_path, tdcsfog_paths.freq,
                     tdcsfog_paths.window_size, tdcsfog_paths.steps)


@hydra.main(config_path='../config', config_name='Config.yaml', version_base='1.3')
def defog_main(config: DictConfig):
    """
    defog main funtion is used to fetch and filter all the data that was obtained from the subjects activities in their
    homes.
    :param config: config parameter is used for accessing the configurations for the specific model.
    """
    current_path = utils.get_original_cwd() + '/'
    defog_paths = config.defog.preprocessing
    dataset = load_data(current_path + defog_paths.metadata, current_path + defog_paths.dataset)
    dataset = dataset.loc[dataset.Valid.eq(True) & dataset.Task.eq(True)]
    dataset = dataset.drop(['Valid', 'Task'], axis=1).reset_index()
    tf_record_writer(dataset, current_path + defog_paths.tf_record_path, defog_paths.freq, defog_paths.window_size,
                     defog_paths.steps)
