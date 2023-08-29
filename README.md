# Freezing of Gait Prediction System

A comprehensive system for predicting freezing of gait (FoG) using machine learning models and enabling real-time predictions through gRPC and WebSocket streaming.

## Table of Contents

- Documentation
  - APIs
    - Explore the API documentation for detailed usage instructions:
      - [pdoc](docs/api/app/index.md)
      - Swagger: run the **api_main.py** and access this is link: http://localhost:8000/docs.
      - ReDoc: run the **api_main.py** and access this is link: http://localhost:8000/redocs.
  
  - [API Main file](docs/api_main.md)
  - [gRPC Stream](docs/grpc_stream/index.md)
  - [Logger](docs/logger_config.md)
  - [Main file](docs/main.md)
  - [Source](docs/src/index.md)
  - [Testing](docs/tests/index.md)
- [License](LICENSE)
- Contact
  - [Email](mailto:aravindh.p201.741@gmail.com).
  - [LinkedIn](https://www.linkedin.com/in/aravindh-pandiyan-80b983145)

## Introduction

The Freezing of Gait Prediction System is designed to predict the occurrence of freezing of gait (FoG) using machine learning models. It provides different components for model construction, training, testing, preprocessing, and real-time streaming of predictions using gRPC and WebSocket.

## Features

- Load different types of machine learning models for FoG prediction.
- Build and train models using both recurrent neural networks (RNN) and convolutional neural networks (CNN).
- Preprocess input data and convert it into suitable formats for training.
- Test the trained models using various metrics to assess their performance.
- Stream real-time predictions using gRPC and WebSocket protocols.
- Interactive API documentation using Swagger and pdoc.

## Installation

Clone the repository:

   ```bash
   git clone https://github.com/AravindhPandiyan/Parkinsons_FOG.git
   cd Parkinsons_FOG
