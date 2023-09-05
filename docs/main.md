Module main
===========
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

Functions
---------

    
`main()`
:   `main` function is the **initiating function** that provides various options from **processing** to **training**
the model and for **inference**.