import json

import requests
from data_processing import process_get

from logger_config import logger as log


def make_api_call(payload: dict, api_type: str) -> str:
    """
    `make_api_call` function is used make a call to the APIs for executing processes.

    Params:
        `payload`: payload is the expected data to be sent to the API.

        `api_type`: api_type is the type of API call that's needed to be made.

    Returns:
        Finally, this function returns the APIs' response.
    """
    log.info("Function Call")
    try:
        with open("config/apis.json") as file:
            apis = json.load(file)

        api = apis["DOMAIN"] + apis["paths"][api_type][1]

        if apis["paths"][api_type][0] == "post":
            response = requests.post(url=api, json=payload)
            response_text = response.json()["detail"]

        else:
            response = requests.get(url=api, json=payload)
            response_text = process_get(response.json(), payload, api_type)

    except requests.exceptions.RequestException as e:
        response_text = f"Error making API call: {e}"
        log.error(response_text)

    except Exception as e:
        response_text = e
        log.error(e)

    return response_text
