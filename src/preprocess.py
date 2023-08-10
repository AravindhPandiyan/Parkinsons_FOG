import dask.dataframe as dd
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import signal as ss


def convert_to_power_spectrums(features: pd.DataFrame, freq: int, wsize: int, step: int) -> np.ndarray:
    """
    Convert to Power Spectrums method is used to obtain the power spectrum of each columnar signal data in the given
    dataframe. Initially, hamming window is applied on the data, to minimize the data's side lobe and improve the
    quality of the data. Next, we make use of FFT to split the data into it's base frequencies. Finally, the power
    spectrum is attained, by computing the portion of a data's power falling within given frequency bins. This entire
    process is completed with the help of the welch method of Scipy's signal processing capabilities.
    :param features: features is the window of dataframe in which each column's power spectrum must be calculated.
    :param freq: freq is the frequency of the data captured.
    :param wsize: wsize is the window size of the data frame
    :param step: step is the step size of the rolling window.
    :return: This function returns the calculated power spectrums.
    """
    pxxs = []
    olap = wsize - step

    for col in features.columns:
        _, pxx = ss.welch(x=features[col], fs=freq, window='hamming', nperseg=wsize, noverlap=olap, scaling='spectrum')
        pxxs.append(pxx)

    pxxs = np.array(pxxs)
    pxxs = pxxs.reshape((-1, pxxs.shape[0]))
    return pxxs


def window_processing(x_win: pd.DataFrame, y_win: pd.DataFrame, freq: int, wsize: int, step: int) -> tuple:
    """
    Window Processing method is used for processing the feature and targets of a given window dataframe.
    :param x_win: x_win is the window dataframe of features.
    :param y_win: y_win is the window dataframe of targets.
    :param freq: freq is the frequency of the data captured.
    :param wsize: wsize is the window size of the data frame
    :param step: step is the step size of the rolling window.
    :return: This function returns a tuple of features and targets.
    """
    x = convert_to_power_spectrums(x_win, freq, wsize, step)

    arr = y_win.values.astype(str)
    unique_rows, counts = np.unique(arr, axis=0, return_counts=True)
    idx_most_frequent_row = np.argmax(counts)
    most_frequent_row = unique_rows[idx_most_frequent_row]
    y_win = pd.DataFrame(most_frequent_row.reshape(1, -1), columns=y_win.columns)
    y = y_win.astype(int).values[0]
    return x, y


def tf_record_writer(data: dd.DataFrame, path: str, freq: int, wsize: int, step: int):
    """
    TF Record Writer function is used to create TFRecords from the given data. The TFRecords are used to improve the
    performance of the model training and handle BIG DATA. This data is usually loaded directly to the GPU.
    :param data: The DataFrame data to be converted to TFRecords.
    :param path: The directory in which the TFRecords data must be stored.
    :param freq: freq is the frequency of the data captured.
    :param wsize: wsize is the window size of the data frame
    :param step: step is the step size of the rolling window.
    """
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


def load_data(meta_path: str, data_path: str) -> dd.DataFrame:
    """
    Load data function is used to filter and fetch the required data from the unprocessed data.
    :param meta_path: meta path is the directory in which the processed metadata is present.
    :param data_path: data path is the directory in which all the data is present.
    :return: Finally, this function returns the filtered data from all the data.
    """
    metadata = pd.read_csv(meta_path)
    dataset_path = list(map(lambda id_: data_path + id_ + '.csv', metadata.Id.unique()))
    dataset = dd.read_csv(dataset_path)
    return dataset
