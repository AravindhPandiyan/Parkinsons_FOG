from src import Inference, Modeling, Preprocessing
from tests import ModelMetrics


def main():
    """
    The Main function is the initiating function that provides various options from processing to training the model.
    """
    processing = Preprocessing()
    modeler = Modeling()
    infer = Inference()
    metrics = ModelMetrics()
    calls = {
        "processes": "\nModeling Options:\n\t1. Preprocessing.\n\t2. Build Models.\n\t3. Train Models.\n\t4. "
                     "Load Models.\n\t5. Test Models.\n\t6. Press any key to Exit.",
        "1": {
            "processes": "\nPre-Processing Options:\n\ta. TDCSFOG RNN Pre-Processing.\n\tb. TDCSFOG CNN Pre-Processing."
                         "\n\tc. DEFOG RNN Pre-Processing.\n\td. DEFOG CNN Pre-Processing.\n\te. "
                         "Press any other key to go back.",
            "a": processing.tdcsfog_rnn_model_preprocessing,
            "b": processing.tdcsfog_cnn_model_preprocessing,
            "c": processing.defog_rnn_model_preprocessing,
            "d": processing.defog_cnn_model_preprocessing
        },
        "2": {
            "processes": "\nModel Building Options:\n\ta. Build TDCSFOG RNN Model.\n\tb. Build TDCSFOG CNN Model."
                         "\n\tc. Build DEFOG RNN Model.\n\td. Build DEFOG CNN Model.\n\te. "
                         "Press any other key to go back.",
            "a": modeler.build_tdcsfog_rnn_model,
            "b": modeler.build_tdcsfog_cnn_model,
            "c": modeler.build_defog_rnn_model,
            "d": modeler.build_defog_cnn_model
        },
        "3": {
            "processes": "\nModel Training Options:\n\ta. Train TDCSFOG RNN Model.\n\tb. Train TDCSFOG CNN Model."
                         "\n\tc. Train DEFOG RNN Model.\n\td. Train DEFOG CNN Model.\n\te. "
                         "Press any other key to go back.",
            "a": modeler.train_tdcsfog_rnn_model,
            "b": modeler.train_tdcsfog_cnn_model,
            "c": modeler.train_defog_rnn_model,
            "d": modeler.train_defog_cnn_model
        },
        "4": {
            "processes": "\nModel Loading Options:\n\ta. Load TDCSFOG RNN Model.\n\tb. Load TDCSFOG CNN Model.\n\tc. "
                         "Load DEFOG RNN Model.\n\td. Load DEFOG CNN Model.\n\te. Press any other key to go back.",
            "a": infer.load_tdcsfog_rnn_model,
            "b": infer.load_tdcsfog_cnn_model,
            "c": infer.load_defog_rnn_model,
            "d": infer.load_defog_cnn_model
        },
        "5": {
            "processes": "\nModel Testing Options:\n\ta. Test TDCSFOG RNN Model.\n\tb. Test TDCSFOG CNN Model.\n\tc. "
                         "Test DEFOG RNN Model.\n\td. Test DEFOG CNN Model.\n\te. Press any other key to go back.",
            "a": metrics.test_tdcsfog_rnn_model,
            "b": metrics.test_tdcsfog_cnn_model,
            "c": metrics.test_defog_rnn_model,
            "d": metrics.test_defog_cnn_model
        }
    }

    while True:
        try:
            print(calls['processes'])
            stage_1 = input('Enter the option number: ')
            print(calls[stage_1]['processes'])

            try:
                stage_2 = input('Enter the option alphabet: ')
                calls[stage_1][stage_2]()

            except KeyError:
                print('\nGoing back...')
                continue

        except (KeyboardInterrupt, KeyError):
            print('\nThank you for Using Parkinson\'s FOG Detection.')
            break


if __name__ == '__main__':
    main()
