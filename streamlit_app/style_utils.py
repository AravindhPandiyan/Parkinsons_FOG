import streamlit as st

from logger_config import logger as log


def set_custom_style():
    """
    `set_custom_style` this funtion is used to set custom style for the generated page from streamlit.
    """

    st.markdown(
        """
        <style>
        .stButton>button {
            width: 150px;
            background-color: #2979FF;
            color: white !important;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: block;
            margin: 0 auto;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #1E69D2;
        }
        .st-expander, .st-form {
            background-color: #f5f5f5;
            padding: 20px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .st-columns {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stRadio>label {
            color: #333;
            font-weight: bold;
            margin-right: 10px;
            display: block;
            margin-bottom: 5px;
        }
        .stRadio > div {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
