import tensorflow as tf


class TFRecordParsers:
    """
    TFRecordParsers class is used to parse the tfrecords file.
    """
    def __init__(self, x_shape, y_shape):
        """
        TFRecordParsers class is used to parse the tfrecords file.
        :param x_shape: x_shape is the shape to which the features should be restored from the tfrecords file.
        :param y_shape: y_shape is the shape to which the targets should be restored from the tfrecords file.
        """
        self.x_shape = list(x_shape)
        self.y_shape = list(y_shape)

    def tfrecord_parser(self, example_proto: tf.Tensor) -> tuple:
        """
        tfrecord_parser methods is used to parse each record of the tfrecord.
        :param example_proto: example_proto is the record to be parsed.
        :return: Finally, the parsed record is divided as feature amd target and returned as a tuple.
        """
        feature_description = {
            'x': tf.io.FixedLenFeature(self.x_shape, tf.float32),
            'y': tf.io.FixedLenFeature(self.y_shape, tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_features['x'], parsed_features['y']
