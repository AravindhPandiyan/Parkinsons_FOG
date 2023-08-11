import tensorflow as tf


def _parse_tdcsfog_tfrecord(example_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Parse tdcsfog tfrecord function is used to access the data in tdcsfog.tfrecords individually with a fixed structure.
    :param example_proto: example proto is a single tensor extracted from the tdcsfog.tfrecords.
    :return: This function returns a tuple of tensors of features and targets from the tdcsfog.tfrecords.
    """
    feature_description = {
        'x': tf.io.FixedLenFeature([165, 3], tf.float32),
        'y': tf.io.FixedLenFeature([3], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    x = parsed_features['x']
    y = parsed_features['y']
    return x, y


def _parse_defog_tfrecord(example_proto: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Parse defog tfrecord function is used to access the data in defog.tfrecords individually with a fixed structure.
    :param example_proto: example proto is a single tensor extracted from the defog.tfrecords.
    :return: This function returns a tuple of tensors of features and targets from the defog.tfrecords.
    """
    feature_description = {
        'x': tf.io.FixedLenFeature([129, 3], tf.float32),
        'y': tf.io.FixedLenFeature([3], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    x = parsed_features['x']
    y = parsed_features['y']
    return x, y
