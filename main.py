from src import Preprocessing, Modeling


def main():
    """
    The Main function is the initiating function that provides various options from processing to training the model.
    """
    processing = Preprocessing()
    modeler = Modeling()
    calls = {
        "processes": "\nModeling Options:\n\t1. Preprocessing.\n\t2. Build Models.\n\t3. Train Models.\n\t4. "
                     "Press any key to Exit.",
        "1": {
            "processes": "\nPre-Processing Options:\n\ta. TDCSFOG Pre-Processing.\n\tb. DEFOG Pre-Processing.\n\tc. "
                         "Press any other key to go back.",
            "a": processing.tdcsfog_preprocessing,
            "b": processing.defog_preprocessing
        },
        "2": {
            "processes": "\nModel Building Options:\n\ta. Build TDCSFOG Model.\n\tb. Build DEFOG Model.\n\tc. "
                         "Press any other key to go back.",
            "a": modeler.build_tdcsfog_model,
            "b": modeler.build_defog_model
        },
        "3": {
            "processes": "\nModel Training Options:\n\ta. Train TDCSFOG Model.\n\tb. Train DEFOG Model.\n\tc. "
                         "Press any other key to go back.",
            "a": modeler.train_tdcsfog_model,
            "b": modeler.train_defog_model
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
