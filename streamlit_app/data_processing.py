from fastapi import Response


def process_get(resp: Response, payload: dict, api_type: str) -> str:
    """
    `process_get` function is used processes the response data received from the particular API.

    Params:
        `resp`: resp is the response from the API.

        `payload`: payload is the expected data to be sent to the API.

        `api_type`: api_type is the type of API call that's needed to be made.

    Returns:
        Finally, this function returns the processes the APIs' response and returns them.
    """
    data = resp.json()
    if resp.status_code == 200:
        if api_type == "TEST_MODEL":
            return f'Mean Average Precision(mAP): {data["map"]}\nArea Under the Curve(AUC): {data["auc"]}'

        else:
            streamer = payload["option"]
            return (
                f'You can do {payload["option"]} Bi-directional streaming using, the below address: '
                f'\n\n{data[f"{streamer}_streaming_address"]}'
            )

    else:
        return data["detail"]
