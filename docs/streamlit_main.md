Module streamlit_app.app
========================
Parkinson's Freezing of Gait Deep Learning Web App.

This script defines a Streamlit web application for performing tasks related
to Deep Learning on Parkinson's Freezing of Gait Data. The app allows users
to configure various parameters such as data types, model architectures, and
API options using expandable sections. User interactions trigger API
calls and display corresponding responses.

The app consists of the following main components:

- Importing necessary modules

- Definition of the `landing_page()` function

- Looping through expandable sections for user interactions

- Displaying API response messages

To run the web app, execute this script. The `landing_page()` function serves
as the entry point for initializing and rendering the Streamlit application.

Modules and utilities used:

- streamlit: For creating interactive web applications

- api_client: For making API calls to backend services

- style_utils: For applying custom styling to the app

- logger_config: For configuring and using logging functionality

Usage:

    streamlit run streamlit_main.py

Functions
---------

    
`landing_page()`
:   `landing_page` is used as the initialization funtion for the streamlit web app.