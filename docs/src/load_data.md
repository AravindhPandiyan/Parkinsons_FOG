Module src.load_data
====================

Classes
-------

`TFRecordParsers(x_shape, y_shape)`
:   `TFRecordParsers` iterate the content of **TFRecords**.
    
    `TFRecordParsers` class is used to parse the **tfrecords** file.
    
    Params:
        `x_shape`: x_shape is the **shape** to which the **features** should be restored from the
        **tfrecords** file.
    
        `y_shape`: y_shape is the **shape** to which the **targets** should be restored from the
        **tfrecords** file.

    ### Methods

    `tfrecord_parser(self, example_proto: tensorflow.python.framework.ops.Tensor) ‑> tuple`
    :   `tfrecord_parser` method is used to **parse** each **record** of the **tfrecord**.
        
        Params:
            `example_proto`: example_proto is the **record** to be parsed.
        
        Returns:
            Finally, the parsed record is divided as feature and target and returned as a tuple.