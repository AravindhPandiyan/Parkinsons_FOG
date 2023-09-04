import asyncio
from collections import deque
from typing import Union

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import JSONResponse

from api.app.controller import InferenceController
from api.app.dependencies import (
    APIResponseModel,
    ArchitectureTypes,
    ModelTypesModel,
    UsableData,
)
from api.app.models import (
    GRPCResponseModel,
    Options,
    SocketPackageModel,
    StreamingOptionModel,
    WebSocketResponseModel,
)
from grpc_stream import GRPCServe, PredictorJob
from logger_config import logger as log

router = APIRouter()
controller = InferenceController()
background_task_status = None


def _background_streaming(address: str):
    """
    `_background_streaming` Sets up a gRPC server-side session as a **background task** within an API call.

    Params:
        `address`: The URL path where the gRPC server-side session will be made available.
    """

    job = PredictorJob(controller)
    grpc_server = GRPCServe(job, address)

    try:
        grpc_server.open_line()
        global background_task_status
        msg = "There was a Connection Timeout as there was no data transfer for 60 seconds."
        log.warning(msg)
        print(f"\n{msg}")
        background_task_status = "completed"

    except Exception as e:
        log.error(e)
        grpc_server.close_line()


@router.post("/load", response_model=APIResponseModel)
async def load_model(load: ModelTypesModel):
    """
    `load_model` is an API route for **loading** different models into the system memory.

    Params:
        `load`: Data received from the user, containing a request for a specific model to be loaded into memory.

    Returns:
        The return values of the function are dependent on the current state of the API.
    """

    try:
        msg = None

        if load.use_data == UsableData.TDCSFOG:
            if load.architecture == ArchitectureTypes.RNN:
                msg = controller.load_tdcsfog_rnn()

            else:
                msg = controller.load_tdcsfog_cnn()

        elif load.use_data == UsableData.DEFOG:
            if load.architecture == ArchitectureTypes.RNN:
                msg = controller.load_defog_rnn()

            else:
                msg = controller.load_defog_cnn()

        if msg:
            log.warning(msg)
            raise HTTPException(
                status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=msg
            )

        else:
            msg = f"{load.use_data} data trained {load.architecture} model has been loaded into memory."

            resp = {"detail": msg}
            return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

    except Exception as e:
        log.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The Server has a Boo Boo.",
        )


@router.websocket("/predict/socket")
async def web_socket_stream(websocket: WebSocket):
    """
    `web_socket_stream` Upgrades the HTTP request response link to a **websocket** for enabling bidirectional
    data transfer for prediction using the model.

    Params:
        `websocket`: The **websocket** parameter passed to the function by the decorator function. It
        is used to control the connection.
    """

    await websocket.accept()
    w_size = controller.window_size
    buffer = deque(maxlen=w_size)
    steps = controller.steps
    count = 0

    try:
        while True:
            try:
                if count < steps or len(buffer) != w_size:
                    package = await asyncio.wait_for(
                        websocket.receive_json(), timeout=60
                    )
                    package = SocketPackageModel(**package)
                    buffer.append([package.AccV, package.AccML, package.AccAP])
                    count += 1

                else:
                    data = pd.DataFrame(buffer)
                    pred = controller.predict(data)
                    pred = pred[0].tolist()

                    if isinstance(pred, list):
                        await websocket.send_json({"prediction": pred})

                    count = 0

            except asyncio.TimeoutError as w:
                msg = "There was a Connection Timeout as there was no data transfer for 60 seconds."
                log.warning(msg + ": " + str(w))
                await websocket.close()
                print(f"\n{msg}")
                break

    except WebSocketDisconnect:
        msg = "A Client just Disconnected."
        print(f"\n{msg}")


@router.get(
    "/predict",
    response_model=Union[APIResponseModel, GRPCResponseModel, WebSocketResponseModel],
)
async def prediction(
    choice: StreamingOptionModel, request: Request, background_tasks: BackgroundTasks
):
    """
    `prediction` is an API route for loading the choices between **Web Socket** and **gRPC** connections for
    **inference streaming**.

    Params:
        `choice`: choice is the data received from the user, containing the type of
        **streaming** requested.

        `request`: request is the parameter passed to the funtion by the decorator funtion, It is used
        for getting the hosting url.

        `background_tasks`: background_tasks is the parameter passed to the funtion by the
        decorator funtion, It is used for running any long-running task or API-freezing task to run in the background.

    Returns:
        The return values of the function are dependent on the state of the API.
    """

    global background_task_status
    if controller.window_size and controller.steps:
        if choice.option == Options.WS:
            msg = str(request.url).replace("http://", "ws://") + "/socket"
            resp = {"WebSocket_streaming_address": msg}
            return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

        else:
            if background_task_status == "running":
                msg = "The gRPC streaming connection is already running."
                log.warning(msg)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

        try:
            domain = request.headers.get("Host").split(":")[0]
            stream_address = domain + ":50051"
            background_task_status = "running"
            background_tasks.add_task(_background_streaming, stream_address)
            resp = {"gRPC_streaming_address": stream_address}
            return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

        except Exception as e:
            log.error(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The Server has a Boo Boo.",
            )

    else:
        msg = "Please First Load the model."
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=msg)
