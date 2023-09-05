Module src.train_model
======================

Functions
---------

    
`fitting(model: keras.engine.training.Model, train_dataset: tensorflow.python.framework.ops.Tensor, val_dataset: tensorflow.python.framework.ops.Tensor, checkpoint_path: str) ‑> keras.engine.training.Model`
:   `fitting` function is used to **train** the model with the given dataset.

    Params:
        `model`: model to be **trained**.
    
        `train_dataset`: train_dataset is used to **train** the model.
    
        `val_dataset`: val_dataset is used to **validate** the **training** of the **model**.
    
        `checkpoint_path`: **checkpoint_path** is folders under **ModelCheckpoint** where the **checkpoint**
        of the **model** is will be **stored**.
    
    Returns:
        Finally, the function returns the **trained** model.