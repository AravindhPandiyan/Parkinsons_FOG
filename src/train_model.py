import json

import tensorflow as tf
from keras.models import Model
from tensorflow_addons.callbacks import TimeStopping


def fitting(
    model: Model, train_dataset: tf.Tensor, val_dataset: tf.Tensor, checkpoint_path: str
) -> Model:
    """
    Fitting function is used to train the model with the given dataset.
    :param model: model to be trained.
    :param train_dataset: train dataset is used to train the model.
    :param val_dataset: val dataset is used to validate the training of the model.
    :param checkpoint_path: checkpoint_path is folders under ModelCheckpoint where the checkpoint of the model is  will
    be stored.
    :return: Finally, the function returns the trained model.
    """
    with open("config/training.json") as file:
        config = json.load(file)

    save_check_point = tf.keras.callbacks.ModelCheckpoint(
        config["checkpoint_loc"] + "/" + checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        save_weights_only=True,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=config["log_loc"], histogram_freq=1, write_graph=True, write_images=True
    )
    time_stopping = TimeStopping(seconds=60 * 60 * 4)
    gpu_count = len(tf.config.list_logical_devices("GPU"))
    gpu_count = gpu_count if gpu_count else 1

    model.fit(
        train_dataset.batch(config["batch_size"] * gpu_count),
        validation_data=val_dataset.batch(config["batch_size"] * gpu_count),
        epochs=config["epochs"],
        verbose=2,
        callbacks=[save_check_point, tensorboard_callback, time_stopping],
    )
    return model
