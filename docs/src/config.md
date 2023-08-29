Module src.config
=================

Classes
-------

`Inference()`
:   `Inference` class **inherits** the **Modeling** class to allow for **loading of model, and prediction**.

    ### Ancestors (in MRO)

    * src.config.Modeling

    ### Descendants

    * api.app.controller.inference_controller.InferenceController
    * tests.test_model.ModelMetrics

    ### Methods

    `load_defog_cnn_model(self) ‑> str | None`
    :   `load_tdcsfog_rnn_model` method is used to load the **DEFOG CNN** model.
        
        Returns:
            If there are **no checkpoints** of the **defog cnn** model it will return a **warning message**,
            else nothing.

    `load_defog_rnn_model(self) ‑> str | None`
    :   `load_tdcsfog_rnn_model` method is used to **load** the **DEFOG RNN** model.
        
        Returns:
            If there are **no checkpoints** of the **defog rnn model** it will return a **warning message**,
            else nothing.

    `load_tdcsfog_cnn_model(self) ‑> str | None`
    :   `load_tdcsfog_rnn_model` method is used to **load** the **TDCSFOG CNN** model.
        
        Returns:
            If there are **no checkpoints** of the **tdcsfog cnn model** it will return a **warning message**,
            else nothing.

    `load_tdcsfog_rnn_model(self) ‑> str | None`
    :   `load_tdcsfog_rnn_model` method is used to **load** the **TDCSFOG RNN** model.
        
        Returns:
            If there are **no checkpoints** of the **tdcsfog rnn model** it will return a **warning message**,
            else nothing.

    `predict_fog(self, data: pd.DataFrame) ‑> list | str`
    :   `predict_fog` method is used for **performing prediction** on the **ML model** that has been **loaded**
        into memory.
        
        Params:
            `data`: data is the series of data from **accelerometer** used for **predicting** if there is a
            **fog detected**.
        
        Returns:
            Finally, it returns the **prediction** list, or if the model is **not loaded** it will return a
            **warning string**.

`Modeling()`
:   `Modeling` is a **configured** class which bring together all the other **functions** and **class** and makes
    use of them to **build amd train the model**.

    ### Descendants

    * api.app.controller.modeling_controller.ModelingController
    * src.config.Inference

    ### Class variables

    `MODEL`
    :

    `MODEL_TYPE`
    :

    `TEST_DATA`
    :

    `TRAIN_DATA`
    :

    `VAL_DATA`
    :

    ### Static methods

    `build_defog_cnn_model(cfg: DictConfig)`
    :   `build_defog_cnn_model` method is used to construct a **cnn model** for **training** on the **defog data**.
        
        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific model.

    `build_defog_rnn_model(cfg: DictConfig)`
    :   `build_defog_rnn_model` method is used to construct a **rnn model** for **training** on the **defog data**.
        
        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific model.

    `build_tdcsfog_cnn_model(cfg: DictConfig)`
    :   `build_tdcsfog_cnn_model` method is used to construct a **cnn model** for **training** on the **tdcsfog data**.
        
        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific model.

    `build_tdcsfog_rnn_model(cfg: DictConfig)`
    :   `build_tdcsfog_rnn_model` method is used to construct a **rnn model** for **training** on the **tdcsfog data**.
        
        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific model.

    `train_defog_cnn_model() ‑> Model | str`
    :   `train_defog_cnn_model` method is used to **train** the **DEFOG CNN** model.
        
        Returns:
            It returns the **model** if the necessary **components** are present to **trains** the **model** and
            return's a **warning string**, if not it will return a **warning message**.

    `train_defog_rnn_model() ‑> Model | str`
    :   `train_defog_rnn_model` method is used to train the **DEFOG RNN** model.
        
        Returns:
            It returns the **model** if the necessary **components** are present to **trains** the **model** and
            return's a **warning string**, if the model is not **constructed** first.

    `train_tdcsfog_cnn_model() ‑> Model | str`
    :   `train_tdcsfog_cnn_model` method is used to train the **TDCSFOG CNN** model.
        
        Returns:
            It returns the **model** if the necessary **components** are present to **train** the **model** and
            return's a **warning string**, if the model is not **constructed** first.

    `train_tdcsfog_rnn_model() ‑> Model | str`
    :   `train_tdcsfog_rnn_model` method is used to train the **TDCSFOG RNN** model.
        
        Returns:
            It returns the **model** if the necessary **components** are present to **train** the **model** and
            return's a **warning string**, if the model is not **constructed** first.

`Preprocessing()`
:   `Preprocessing` is a **configured** class that bring together all the other functions and class and makes use of
    them to **process the data**.

    ### Descendants

    * api.app.controller.preprocessing_controller.PreprocessorController

    ### Static methods

    `defog_cnn_model_preprocessing(cfg: DictConfig)`
    :   `defog_rnn_model_preprocessing` method is used to **fetch** and **filter** all the data that was obtained
        from the **subjects activities** in their **homes** for **cnn** a model.
        
        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific process
            types **processing**.

    `defog_rnn_model_preprocessing(cfg: DictConfig)`
    :   `defog_rnn_model_preprocessing` method is used to **fetch** and **filter** all the data that was obtained
        from the **subjects activities** in their **homes** for **rnn** a model.
        
        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific process
            types **processing**.
        
        Returns:

    `tdcsfog_cnn_model_preprocessing(cfg: DictConfig)`
    :   `tdcsfog_cnn_model_preprocessing` method is used to **fetch** and **filter** all the data that was
        tested in **lab conditions** for **cnn** a model.
        
        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific process
            types **processing**.

    `tdcsfog_rnn_model_preprocessing(cfg: DictConfig)`
    :   `tdcsfog_rnn_model_preprocessing` method is used to **fetch** and **filter** all the data that was
        tested in **lab conditions** for **rnn** model.
        
        Params:
            `cfg`: cfg parameter is used for accessing the **configurations** for the specific process
            types **processing**.