Module src.preprocess
=====================

Classes
-------

`WindowWriter(window_size: int = 10, steps: int = 1, freq: int = 100, model_type: str = 'CNN')`
:   `WindowWriter` class creates **TFRecords**.
    
    `WindowWriter` class is used for converting the data into **tfrecords**.
    
    Params:
        `window_size`: window_size is the window size of the data frame.
    
        `steps`: steps is the step size of the **rolling window**.
    
        `freq`: freq is the **frequency** of the data captured.
    
        `model_type`: model_type it is to mention the type of model to opt into a particular type of
        processing of data.

    ### Static methods

    `load_csv_data(meta_path: str, data_path: str) ‑> dask.dataframe.core.DataFrame`
    :   `load_csv_data` method is used to **filter and fetch** the required data from the **unprocessed csv data**.
        Params:
            `meta_path`: meta_path is the **directory** in which the **processed metadata** is present.
        
            `data_path`: data_path is the **directory** in which all the data is present.
        
        Returns:
            Finally, this function returns the **filtered data** from all the data.

    ### Methods

    `tf_record_writer(self, data: dask.dataframe.core.DataFrame, path: str)`
    :   `tf_record_writer` method is used to create **TFRecords** from the given data. The **TFRecords** are used to
        **improve the performance** of the model training and handle **BIG DATA**. This data is usually loaded
        directly to the **GPU**.
        
        Params:
            `data`: The DataFrame data to be converted to **TFRecords**.
        
            `path`: The **directory** in which the **TFRecords** data must be stored.
        
        Returns:
            Finally, this function returns the **length of the tfrecord dataset**.

    `window_processing(self, x_win: pandas.core.frame.DataFrame, y_win: pandas.core.frame.DataFrame =    0  1  2
    0  0  0  0) ‑> tuple[numpy.ndarray, numpy.ndarray]`
    :   `window_processing` method is used for **processing** the **feature and targets** of a given
        **window dataframe**.
        
        Params:
            `x_win`: x_win is the window dataframe of features.
        
            `y_win`: y_win is the window dataframe of targets.
        
        Returns:
            This function returns a tuple of **flattened features and majority target**.