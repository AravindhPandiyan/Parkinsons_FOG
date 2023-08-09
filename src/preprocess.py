import dask.dataframe as dd
import hydra
import numpy as np
import pandas as pd
import tensorflow as tf

from hydra import utils
from scipy import signal as ss


def convert_to_power_spectrums(features, freq, wsize, step):
    pxxs = []
    olap = wsize - step

    for col in features.columns:
        _, pxx = ss.welch(x=features[col], fs=freq, window='hamming', nperseg=wsize, noverlap=olap, scaling='spectrum')
        pxxs.append(pxx)

    pxxs = np.array(pxxs)
    pxxs = pxxs.reshape((-1, pxxs.shape[0]))
    return pxxs


def window_processing(x_win, y_win, freq, wsize, step):
    x = convert_to_power_spectrums(x_win, freq, wsize, step)

    arr = y_win.values.astype(str)
    unique_rows, counts = np.unique(arr, axis=0, return_counts=True)
    idx_most_frequent_row = np.argmax(counts)
    most_frequent_row = unique_rows[idx_most_frequent_row]
    y_win = pd.DataFrame(most_frequent_row.reshape(1, -1), columns=y_win.columns)
    y = y_win.astype(int).values[0]
    return x, y


def tf_record_writer(data, path, freq, wsize, step):
    data = data.fillna(0)

    with tf.io.TFRecordWriter(path) as writer:
        for partition in data.partitions:
            features = partition.iloc[:, 1:-3].compute()
            target = partition.iloc[:, -3:].compute()

            for win_start in range(0, features.shape[0], step):
                x_win = features.iloc[win_start: win_start + wsize, :]
                y_win = target.iloc[win_start: win_start + wsize, :]

                if x_win.shape[0] == wsize:
                    x, y = window_processing(x_win, y_win, freq, wsize, step)

                    x = x.tobytes()
                    record_bytes = tf.train.Example(features=tf.train.Features(feature={
                        "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x])),
                        "y": tf.train.Feature(int64_list=tf.train.Int64List(value=y)),
                    })).SerializeToString()
                    writer.write(record_bytes)


def load_data(meta_path, data_path):
    metadata = pd.read_csv(meta_path)
    dataset_path = list(map(lambda id_: data_path + id_ + '.csv', metadata.Id.unique()))
    dataset = dd.read_csv(dataset_path)
    return dataset


@hydra.main(config_path='../config', config_name='Config.yaml', version_base='1.3')
def tdcsfog_main(config):
    current_path = utils.get_original_cwd() + '/'
    tdcsfog_paths = config.preprocessing.tdcsfog
    dataset = load_data(current_path + tdcsfog_paths.metadata, current_path + tdcsfog_paths.dataset)
    tf_record_writer(dataset, current_path + tdcsfog_paths.tf_record_path, tdcsfog_paths.freq,
                     tdcsfog_paths.window_size, tdcsfog_paths.steps)


@hydra.main(config_path='../config', config_name='Config.yaml', version_base='1.3')
def defog_main(config):
    current_path = utils.get_original_cwd() + '/'
    defog_paths = config.preprocessing.defog
    dataset = load_data(current_path + defog_paths.metadata, current_path + defog_paths.dataset)
    dataset = dataset.loc[dataset.Valid.eq(True)]
    dataset = dataset.drop(['Valid', 'Task'], axis=1).reset_index()
    tf_record_writer(dataset, current_path + defog_paths.tf_record_path, defog_paths.freq, defog_paths.window_size,
                     defog_paths.steps)
