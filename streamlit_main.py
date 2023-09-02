"""
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

  streamlit run streamlit_app/streamlit_main.py

"""
import streamlit as st

from logger_config import logger as log
from streamlit_app.api_client import make_api_call
from streamlit_app.style_utils import set_custom_style


def landing_page():
    """
    `landing_page` is used as the initialization funtion for the streamlit web app.
    """
    log.info("Funtion Call")
    set_custom_style()
    st.title("Deep Learning on Parkinson's Freezing of Gait Data.")

    expander_names = [
        "Preprocessing",
        "Build Model",
        "Train Model",
        "Load Model",
        "Test Model",
        "Streamers",
    ]

    additional_details = ""

    for key, expander_names in enumerate(expander_names):
        api_json = {}
        with st.expander(expander_names):
            if expander_names != "Streamers":
                with st.form(expander_names):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Data Type")
                        api_json["use_data"] = st.radio("", ["TDCSFOG", "DEFOG"])

                    with col2:
                        st.markdown("### Model Type")
                        api_json["architecture"] = st.radio("", ["RNN", "CNN"])

                    submitted = st.form_submit_button("Submit")

            else:
                with st.form(expander_names):
                    api_json["option"] = st.radio(
                        "", ["WebSocket", "gRPC"], horizontal=True
                    )
                    submitted = st.form_submit_button("Submit")

        if submitted:
            additional_details = make_api_call(
                api_json, expander_names.upper().replace(" ", "_")
            )

    st.markdown("### API Response Message")
    st.text_area("Response:", value=additional_details)


if __name__ == "__main__":
    landing_page()
