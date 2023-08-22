import asyncio
from collections import deque

import pandas as pd
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from api.app.controller import InferenceController

router = APIRouter()
infer = InferenceController()


@router.post('/load/tdcsfog/rnn', status_code=200)
async def load_tdcsfog_rnn():
    """
    load_tdcsfog_rnn is api router path for loading tdcsfog data trained rnn model into memory.
    """
    try:
        msg = infer.load_tdcsfog_rnn()

        if msg:
            return JSONResponse(status_code=424, content={"details": msg})

    except Exception:
        raise HTTPException(status_code=500, detail='The Server has a Boo Boo.')


@router.post('/load/tdcsfog/cnn', status_code=200)
async def load_tdcsfog_cnn():
    """
    load_tdcsfog_cnn is api router path for loading tdcsfog data trained cnn model into memory.
    """
    try:
        msg = infer.load_tdcsfog_cnn()

        if msg:
            return JSONResponse(status_code=424, content={"details": msg})

    except Exception:
        raise HTTPException(status_code=500, detail='The Server has a Boo Boo.')


@router.post('/load/defog/rnn', status_code=200)
async def load_defog_rnn():
    """
    load_defog_rnn is api router path for loading defog data trained rnn model into memory.
    """
    try:
        msg = infer.load_defog_rnn()

        if msg:
            return JSONResponse(status_code=424, content={"details": msg})

    except Exception:
        raise HTTPException(status_code=500, detail='The Server has a Boo Boo.')


@router.post('/load/defog/cnn', status_code=200)
async def load_defog_cnn():
    """
    load_defog_rnn is api router path for loading defog data trained cnn model into memory.
    """
    try:
        msg = infer.load_defog_cnn()

        if msg:
            return JSONResponse(status_code=424, content={"details": msg})

    except Exception:
        raise HTTPException(status_code=500, detail='The Server has a Boo Boo.')


@router.websocket('/predict')
async def predict(websocket: WebSocket):
    """
    predict function is used for upgrading the http request responser link to a websocket for allowing bidirectional
    data transfer for prediction using the model.
    :param websocket:
    """
    await websocket.accept()
    w_size = infer.infer.window_size
    buffer = deque(maxlen=w_size)
    steps = infer.infer.steps
    count = 0

    try:
        while True:
            try:
                if count < steps:
                    package = await asyncio.wait_for(websocket.receive_json(), timeout=60)
                    buffer.append(package['data'])
                    count += 1

                elif len(buffer) != w_size:
                    package = await asyncio.wait_for(websocket.receive_json(), timeout=60)
                    buffer.append(package['data'])
                    count += 1

                else:
                    data = pd.DataFrame(buffer)
                    pred = infer.predict(data)
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
