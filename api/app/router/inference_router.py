from fastapi import APIRouter, HTTPException
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
