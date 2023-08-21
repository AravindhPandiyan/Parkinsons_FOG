from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.app.controller import ModelingController

router = APIRouter()
modeler = ModelingController()


@router.post('/build/tdcsfog/rnn', status_code=201)
async def build_tdcsfog_rnn():
    """
    build_tdcsfog_rnn is api router path for building a rnn model for tdcsfog data.
    """
    try:
        modeler.build_tdcsfog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/build/tdcsfog/cnn', status_code=201)
async def build_tdcsfog_cnn():
    """
    build_tdcsfog_cnn is api router path for building a cnn model for tdcsfog data.
    """
    try:
        modeler.build_tdcsfog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/build/defog/rnn', status_code=201)
async def build_defog_rnn():
    """
    build_defog_rnn is api router path for building a rnn model for defog data.
    """
    try:
        modeler.build_defog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/build/defog/cnn', status_code=201)
async def build_defog_cnn():
    """
    build_defog_cnn is api router path for building a cnn model for defog data.
    """
    try:
        modeler.build_defog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/tdcsfog/rnn', status_code=200)
async def train_tdcsfog_rnn():
    """
    train_tdcsfog_rnn is api router path for training the rnn model with tdcsfog data.
    """
    try:
        msg = modeler.train_tdcsfog_rnn()

        if isinstance(msg, str):
            return JSONResponse(status_code=424, content={"details": msg})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/tdcsfog/cnn', status_code=200)
async def train_tdcsfog_cnn():
    """
    train_tdcsfog_cnn is api router path for training the cnn model with tdcsfog data.
    """
    try:
        msg = modeler.train_tdcsfog_cnn()

        if isinstance(msg, str):
            return JSONResponse(status_code=424, content={"details": msg})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/defog/rnn', status_code=200)
async def train_defog_rnn():
    """
    train_defog_rnn is api router path for training the rnn model with defog data.
    """
    try:
        msg = modeler.train_defog_rnn()

        if isinstance(msg, str):
            return JSONResponse(status_code=424, content={"details": msg})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/defog/cnn', status_code=200)
async def train_defog_cnn():
    """
    train_defog_rnn is api router path for training the cnn model with defog data.
    """
    try:
        msg = modeler.train_defog_cnn()

        if isinstance(msg, str):
            return JSONResponse(status_code=424, content={"details": msg})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
