import tensorflow as tf


class TFRecordParsers:
    def __init__(self, x_shape, y_shape):
        self.x_shape = list(x_shape)
        self.y_shape = list(y_shape)

    def tfrecord_parser(self, example_proto: tf.Tensor):
        feature_description = {
            'x': tf.io.FixedLenFeature(self.x_shape, tf.float32),
            'y': tf.io.FixedLenFeature(self.y_shape, tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_features['x'], parsed_features['y']
