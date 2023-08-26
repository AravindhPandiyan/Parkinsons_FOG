import asyncio
from collections import deque
from typing import Union

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from api.app.controller import InferenceController
from api.app.dependencies import APIResponseModel, ArchitectureTypes, ModelTypesModel, UsableData
from api.app.models import GRPCResponseModel, Options, SocketPackageModel, StreamingOptionModel, WebSocketResponseModel
from grpc_stream import GRPCServe, PredictorJob

router = APIRouter()
controller = InferenceController()
background_task_status = None


def background_streaming(address):
    job = PredictorJob(controller)
    grpc_server = GRPCServe(job, address)

    try:
        grpc_server.open_line()
        global background_task_status
        print('\nThere was a Connection Timeout as there was no data transfer for 60 seconds.')
        background_task_status = 'completed'

    except Exception:
        grpc_server.close_line()


@router.post('/load', response_model=APIResponseModel)
async def load_model(load: ModelTypesModel):
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
            return JSONResponse(status_code=status.HTTP_424_FAILED_DEPENDENCY, content=resp)

        else:
            msg = f'{load.use_data} data trained {load.architecture} model has been loaded into memory.'
            resp = {'detail': msg}
            return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The Server has a Boo Boo.')


@router.websocket('/predict/socket')
async def web_socket_stream(websocket: WebSocket):
    """
    predict function is used for upgrading the http request responser link to a websocket for allowing bidirectional
    data transfer for prediction using the model.
    :param websocket:
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
                    package = await asyncio.wait_for(websocket.receive_json(), timeout=60)
                    package = SocketPackageModel(**package)
                    buffer.append([package.AccV, package.AccML, package.AccAP])
                    count += 1

                else:
                    data = pd.DataFrame(buffer)
                    pred = controller.predict(data)
                    pred = pred[0].tolist()

                    if isinstance(pred, list):
                        await websocket.send_json({'prediction': pred})

                    count = 0

            except asyncio.TimeoutError:
                await websocket.close()
                print('\nThere was a Connection Timeout as there was no data transfer for 60 seconds.')
                break

    except WebSocketDisconnect:
        print('\nA Client just Disconnected.')


@router.get('/predict', response_model=Union[APIResponseModel, GRPCResponseModel, WebSocketResponseModel])
async def prediction(choice: StreamingOptionModel, request: Request, background_tasks: BackgroundTasks):
    global background_task_status

    if choice.option == Options.WebSocket:
        msg = str(request.url).replace("http://", "ws://") + '/socket'
        resp = {"web_socket_address": msg}
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

    else:
        if background_task_status == 'running':
            msg = f"The gRPC streaming connection is already running."
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    try:
        domain = request.headers.get("Host").split(':')[0]
        stream_address = domain + ':50051'
        background_task_status = 'running'
        background_tasks.add_task(background_streaming, stream_address)
        resp = {"gRPC_stream_address": stream_address}
        return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The Server has a Boo Boo.')
