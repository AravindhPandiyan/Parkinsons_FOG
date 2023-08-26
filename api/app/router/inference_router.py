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

router = APIRouter()
controller = InferenceController()
background_task_status = None


def _background_streaming(address):
    """
    _background_streaming is a private funtion used for setting up gRPC server-side session as a background task in an
    API call.
    :param address: address is the url path in which the gRPC server-side session will be made available.
    """
    job = PredictorJob(controller)
    grpc_server = GRPCServe(job, address)

    try:
        grpc_server.open_line()
        global background_task_status
        print(
            "\nThere was a Connection Timeout as there was no data transfer for 60 seconds."
        )
        background_task_status = "completed"

    except Exception:
        grpc_server.close_line()


@router.post("/load", response_model=APIResponseModel)
async def load_model(load: ModelTypesModel):
    """
    load_model is an API route for loading the different model's into the system memory.
    :param load: load is the data received from the user, containing request of a specific model to be loaded into
    memory.
    :return: The return values of the function is dependent on the state of API.
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
            resp = {"detail": msg}
            return JSONResponse(
                status_code=status.HTTP_424_FAILED_DEPENDENCY, content=resp
            )

        else:
            msg = f"{load.use_data} data trained {load.architecture} model has been loaded into memory."
            resp = {"detail": msg}
            return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The Server has a Boo Boo.",
        )


@router.websocket("/predict/socket")
async def web_socket_stream(websocket: WebSocket):
    """
    web_socket_stream function is used for upgrading the http request responser link to a websocket for allowing
    bidirectional data transfer for prediction using the model.
    :param websocket: websocket is the parameter passed to the funtion by the decorator funtion, It is used for getting
    the controlling the connection.
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

            except asyncio.TimeoutError:
                await websocket.close()
                print(
                    "\nThere was a Connection Timeout as there was no data transfer for 60 seconds."
                )
                break

    except WebSocketDisconnect:
        print("\nA Client just Disconnected.")


@router.get(
    "/predict",
    response_model=Union[APIResponseModel, GRPCResponseModel, WebSocketResponseModel],
)
async def prediction(
    choice: StreamingOptionModel, request: Request, background_tasks: BackgroundTasks
):
    """
    prediction is an API route for loading the choices between Web Socket and gRPC connections for inference streaming.
    :param choice: choice is the data received from the user, containing the type of streaming requested.
    :param request: request is the parameter passed to the funtion by the decorator funtion, It is used for getting the
    hosting url.
    :param background_tasks: background_tasks is the parameter passed to the funtion by the decorator funtion, It is
    used for running any long-running task or API freezing task to run in the background.
    :return: The return values of the function is dependent on the state of API.
    """
    global background_task_status

    if choice.option == Options.WebSocket:
        msg = str(request.url).replace("http://", "ws://") + "/socket"
        resp = {"web_socket_address": msg}
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

    else:
        if background_task_status == "running":
            msg = "The gRPC streaming connection is already running."
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    try:
        domain = request.headers.get("Host").split(":")[0]
        stream_address = domain + ":50051"
        background_task_status = "running"
        background_tasks.add_task(_background_streaming, stream_address)
        resp = {"gRPC_stream_address": stream_address}
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The Server has a Boo Boo.",
        )
