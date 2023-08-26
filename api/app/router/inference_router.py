import asyncio
from collections import deque

import pandas as pd
from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from api.app.controller import InferenceController
from api.app.dependencies import APIResponseModel, ArchitectureTypes, ModelTypesModel, UsableData
from api.app.models import SocketPackageModel

router = APIRouter()
controller = InferenceController()


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

# @router.get('/predict', status_code=status.HTTP_200_OK)
# async def prediction():
