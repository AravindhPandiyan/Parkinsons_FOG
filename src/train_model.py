import json

import tensorflow as tf
from keras.models import Model
from tensorflow_addons.callbacks import TimeStopping


def fitting(model: Model, dataset: tf.Tensor) -> Model:
    """
    Fitting function is used to train the model with the given dataset.
    :param model: model to be trained.
    :param dataset: dataset is used to train the model.
    :return: Finally, the function returns the trained model.
    """
    with open('config/training.json') as file:
        config = json.load(file)

    save_check_point = tf.keras.callbacks \
        .ModelCheckpoint(config['checkpoint_loc'], monitor="val_precision", save_best_only=True, mode='max',
                         save_weights_only=True)
    tensorboard_callback = tf.keras.callbacks \
        .TensorBoard(log_dir=config['log_loc'], histogram_freq=1, write_graph=True, write_images=True)
    time_stopping = TimeStopping(seconds=60 * 60 * 4, verbose=1)
    early_stopping_callback = tf.keras.callbacks \
        .EarlyStopping(monitor="val_precision", patience=11, mode='max', restore_best_weights=True)

    model.fit(dataset.batch(config['batch_size']), epochs=config['epochs'], validation_split=0.2,
              callbacks=[save_check_point, tensorboard_callback, time_stopping, early_stopping_callback], verbose=1)
    return model
