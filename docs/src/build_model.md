Module src.build_model
======================

Classes
-------

`ConstructCNN(drop_rate: float = 0.2, fun_type: str = 'relu')`
:   `ConstructCNN` is **constructs CNN model**.
    
    `ConstructCNN` class is used for building **CNN** a based model.
    
    Params:
        `drop_rate`: drop_rate is used to mention the **rate of dropout layer's**.
    
        `fun_type`: fun_type is the type of **activation function**.

    ### Methods

    `build_model(self, data: tensorflow.python.framework.ops.Tensor, size: tuple, nodes: omegaconf.dictconfig.DictConfig) ‑> keras.engine.training.Model`
    :   `build_model` method is using the properties of the class to build the model.
        
        Params:
            `data`: data is used to find it's **mean and variance** to **normalize** it.
        
            `size`: size is used to mention the **input shape** of the data.
        
            `nodes`: nodes value is used for mentioning the number of **neurons** in each layer.
        
        Returns:
            Finally, this method returns the constructed model.

`ConstructRNN(drop_rate: float = 0.2, fun_type: str = 'tanh')`
:   `ConstructRNN` is **constructs RNM model**.
    
    `ConstructRNN` class is used for building **RNN** based model.
    
    Params:
        `drop_rate`: drop_rate is used to mention the **rate of dropout layer's**.
    
        `fun_type`: fun_type is the type of **activation function**.

    ### Methods

    `build_model(self, data: tensorflow.python.framework.ops.Tensor, size: tuple, nodes: int) ‑> keras.engine.training.Model`
    :   `build_model` method is uses the properties of the class to **build** the model.
        
        Params:
            `data`: data is used to find it's **mean and variance** to **normalize** it.
        
            `size`: size is used to mention the **input shape** of the data.
        
            `nodes`: nodes value is used for mentioning the number of **neurons** in each layer.
        
        Returns:
            Finally, this method returns the constructed model.