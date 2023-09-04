from typing import Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal as ss

from logger_config import logger as log


class WindowWriter:
    """
    `WindowWriter` class creates **TFRecords**.
    """

    def __init__(
        self,
        window_size: int = 10,
        steps: int = 1,
        freq: int = 100,
        model_type: str = "CNN",
    ):
        """
        `WindowWriter` class is used for converting the data into **tfrecords**.

        Params:
            `window_size`: window_size is the window size of the data frame.

            `steps`: steps is the step size of the **rolling window**.

            `freq`: freq is the **frequency** of the data captured.

            `model_type`: model_type it is to mention the type of model to opt into a particular type of
            processing of data.
        """

        self.wsize = window_size
        self.steps = steps
        self.freq = freq
        self.m_type = model_type
        self._olap = self.wsize - self.steps

    def _convert_to_power_spectrums(self, features: pd.DataFrame) -> np.ndarray:
        """
        `_convert_to_power_spectrums` method is used to obtain the **power spectrum** of each **columnar signal data**
        in the given dataframe. Initially, **hamming window** is applied on the data to minimize the data's
        **side lobe** and improve the quality of the data. Next, we make use of **FFT** to split the data into
         its base **frequencies**. Finally, the **power spectrum** is attained by computing the portion of
        data's power falling within given **frequency bins**. This entire process is completed with the help of the
        **welch** method of **Scipy's signal processing** capabilities.

        Params:
            `features`: features is the window of dataframe in which each column's **power spectrum** must
            be calculated.

        Returns:
            This function returns the **calculated power spectrum's**.
        """
        pxxs = []

        for col in features.columns:
            _, pxx = ss.welch(
                x=features[col],
                fs=self.freq,
                window="hamming",
                nperseg=self.wsize,
                noverlap=self._olap,
                scaling="spectrum",
            )
            pxxs.append(pxx)

        pxxs = np.array(pxxs)
        pxxs = pxxs.reshape((-1, pxxs.shape[0]))
        return pxxs

    def window_processing(
        self, x_win: pd.DataFrame, y_win: pd.DataFrame = pd.DataFrame([[0, 0, 0]])
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        `window_processing` method is used for **processing** the **feature and targets** of a given
        **window dataframe**.

        Params:
            `x_win`: x_win is the window dataframe of features.

            `y_win`: y_win is the window dataframe of targets.

        Returns:
            This function returns a tuple of **flattened features and majority target**.
        """
        if self.m_type == "RNN":
            x = self._convert_to_power_spectrums(x_win)

        else:
            x = x_win.values

        arr = y_win.values.astype(str)
        unique_rows, counts = np.unique(arr, axis=0, return_counts=True)
        idx_most_frequent_row = np.argmax(counts)
        most_frequent_row = unique_rows[idx_most_frequent_row]
        y_win = pd.DataFrame(most_frequent_row.reshape(1, -1), columns=y_win.columns)
        y = y_win.astype(int).values[0]
        return x.flatten(), y

    def tf_record_writer(self, data: dd.DataFrame, path: str):
        """
        `tf_record_writer` method is used to create **TFRecords** from the given data. The **TFRecords** are used to
        **improve the performance** of the model training and handle **BIG DATA**. This data is usually loaded
        directly to the **GPU**.

        Params:
            `data`: The DataFrame data to be converted to **TFRecords**.

            `path`: The **directory** in which the **TFRecords** data must be stored.

        Returns:
            Finally, this function returns the **length of the tfrecord dataset**.
        """

        data = data.fillna(0)
        size = 0
        with tf.io.TFRecordWriter(path) as writer:
            for partition in data.partitions:
                features = partition.iloc[:, 1:-3].compute()
                target = partition.iloc[:, -3:].compute()

                for win_start in range(0, features.shape[0], self.steps):
                    x_win = features.iloc[win_start : win_start + self.wsize, :]
                    y_win = target.iloc[win_start : win_start + self.wsize, :]

                    if x_win.shape[0] == self.wsize:
                        x, y = self.window_processing(x_win, y_win)
                        record_bytes = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    "x": tf.train.Feature(
                                        float_list=tf.train.FloatList(value=x)
                                    ),
                                    "y": tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=y)
                                    ),
                                }
                            )
                        ).SerializeToString()
                        writer.write(record_bytes)
                        size += 1

        return size

    @staticmethod
    def load_csv_data(meta_path: str, data_path: str) -> dd.DataFrame:
        """
        `load_csv_data` method is used to **filter and fetch** the required data from the **unprocessed csv data**.
        Params:
            `meta_path`: meta_path is the **directory** in which the **processed metadata** is present.

            `data_path`: data_path is the **directory** in which all the data is present.

        Returns:
            Finally, this function returns the **filtered data** from all the data.

        """

        metadata = pd.read_csv(meta_path)
        dataset_path = list(
            map(lambda id_: data_path + id_ + ".csv", metadata.Id.unique())
        )
        dataset = dd.read_csv(dataset_path)
        return dataset
