"""
Parkinson's FOG Detection Application
--------------------------------------

This module contains the main functionality for a Parkinson's FOG (Freezing of Gait) detection application.
It provides options for preprocessing, modeling, training, testing, and inference using various models.
The application also includes a private `_grpc_test` function for simultaneous testing of server and client.

Usage:

- Run this script directly to initiate the Parkinson's FOG Detection application.

- The application provides various options for preprocessing, modeling, training, testing, and inference.

Modules:

- `concurrent.futures`: Provides high-level interface for asynchronously executing functions.

- `src.Inference`: Contains the `Inference` class for model inference.

- `src.Modeling`: Contains the `Modeling` class for model building and training.

- `src.Preprocessing`: Contains the `Preprocessing` class for data preprocessing.

- `tests.ModelMetrics`: Contains the `ModelMetrics` class for model testing.

- `tests.data_streamer`: Contains the function `data_streamer` for streaming data to the client.

- `tests.server`: Contains the function `server` for serving gRPC inference requests.

Functions:

- `_grpc_test`: Private function to start simultaneous thread execution of server and client code for testing.

- `main`: Initiating function that provides options for preprocessing, modeling, training, testing, and inference.

"""

import concurrent.futures

from logger_config import logger as log
from src import Inference, Modeling, Preprocessing
from tests import ModelMetrics, data_streamer, server


def _grpc_test(inference):
    """
    `_grpc_test` is a **private funtion** to start **simultaneous thread execution** of the **server-side** and
    **client-side** code for **testing** purpose.

    Params:
        inference: inference is to make use of the existing inference instance in the server-side of the code.
    """
    log.info("Function Call")

    with concurrent.futures.ThreadPoolExecutor() as exe:
        exe.submit(server, inference)
        exe.submit(data_streamer)


def main():
    """
    `main` function is the **initiating function** that provides various options from **processing** to **training**
    the model and for **inference**.
    """
    log.info("Function Call")
    processing = Preprocessing()
    modeler = Modeling()
    infer = Inference()
    metrics = ModelMetrics()

    calls = {
        "processes": "\nModeling Options:\n\t1. Preprocessing.\n\t2. Build Models.\n\t3. Train Models.\n\t4. "
        "Load Models.\n\t5. Test Models.\n\t6. Stream For Inference Testing.\n\t7. Press any key to Exit.",
        "1": {
            "processes": "\nPre-Processing Options:\n\ta. TDCSFOG RNN Pre-Processing.\n\tb. TDCSFOG CNN Pre-Processing."
            "\n\tc. DEFOG RNN Pre-Processing.\n\td. DEFOG CNN Pre-Processing.\n\te. "
            "Press any other key to go back.",
            "a": processing.tdcsfog_rnn_model_preprocessing,
            "b": processing.tdcsfog_cnn_model_preprocessing,
            "c": processing.defog_rnn_model_preprocessing,
            "d": processing.defog_cnn_model_preprocessing,
        },
        "2": {
            "processes": "\nModel Building Options:\n\ta. Build TDCSFOG RNN Model.\n\tb. Build TDCSFOG CNN Model."
            "\n\tc. Build DEFOG RNN Model.\n\td. Build DEFOG CNN Model.\n\te. "
            "Press any other key to go back.",
            "a": modeler.build_tdcsfog_rnn_model,
            "b": modeler.build_tdcsfog_cnn_model,
            "c": modeler.build_defog_rnn_model,
            "d": modeler.build_defog_cnn_model,
        },
        "3": {
            "processes": "\nModel Training Options:\n\ta. Train TDCSFOG RNN Model.\n\tb. Train TDCSFOG CNN Model."
            "\n\tc. Train DEFOG RNN Model.\n\td. Train DEFOG CNN Model.\n\te. "
            "Press any other key to go back.",
            "a": modeler.train_tdcsfog_rnn_model,
            "b": modeler.train_tdcsfog_cnn_model,
            "c": modeler.train_defog_rnn_model,
            "d": modeler.train_defog_cnn_model,
        },
        "4": {
            "processes": "\nModel Loading Options:\n\ta. Load TDCSFOG RNN Model.\n\tb. Load TDCSFOG CNN Model.\n\tc. "
            "Load DEFOG RNN Model.\n\td. Load DEFOG CNN Model.\n\te. Press any other key to go back.",
            "a": infer.load_tdcsfog_rnn_model,
            "b": infer.load_tdcsfog_cnn_model,
            "c": infer.load_defog_rnn_model,
            "d": infer.load_defog_cnn_model,
        },
        "5": {
            "processes": "\nModel Testing Options:\n\ta. Test TDCSFOG RNN Model.\n\tb. Test TDCSFOG CNN Model.\n\tc. "
            "Test DEFOG RNN Model.\n\td. Test DEFOG CNN Model.\n\te. Press any other key to go back.",
            "a": metrics.test_tdcsfog_rnn_model,
            "b": metrics.test_tdcsfog_cnn_model,
            "c": metrics.test_defog_rnn_model,
            "d": metrics.test_defog_cnn_model,
        },
        "6": _grpc_test,
    }

    while True:
        try:
            print(calls["processes"])
            stage_1 = input("Enter the option number: ")
            if stage_1 != "6":
                print(calls[stage_1]["processes"])

                try:
                    stage_2 = input("Enter the option alphabet: ")
                    calls[stage_1][stage_2]()

                except KeyError as i:
                    msg = "Going back..."
                    log.info(msg + ": " + str(i))
                    print(f"\n{msg}")
                    continue

            else:
                if infer.steps and infer.window_size:
                    calls[stage_1](infer)

                else:
                    msg = "Please First Load the model."
                    log.warning(msg)
                    print(f"\n{msg}")

        except (KeyboardInterrupt, KeyError) as i:
            msg = "Thank you for Using Parkinson's FOG Detection."
            log.info(msg + ": " + str(i))
            print(f"\n{msg}")
            break


if __name__ == "__main__":
    main()
