from fastapi import APIRouter, HTTPException

from api.app.controller import InferenceController

router = APIRouter()
infer = InferenceController()


@router.post('/load/tdcsfog/rnn', status_code=200)
async def load_tdcsfog_rnn():
    """
    load_tdcsfog_rnn is api router path for loading tdcsfog data trained rnn model into memory.
    """
    try:
        infer.load_tdcsfog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/load/tdcsfog/cnn', status_code=200)
async def load_tdcsfog_cnn():
    """
    load_tdcsfog_cnn is api router path for loading tdcsfog data trained cnn model into memory.
    """
    try:
        infer.load_tdcsfog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/load/defog/rnn', status_code=200)
async def load_defog_rnn():
    """
    load_defog_rnn is api router path for loading defog data trained rnn model into memory.
    """
    try:
        infer.load_defog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/load/defog/cnn', status_code=200)
async def load_defog_cnn():
    """
    load_defog_rnn is api router path for loading defog data trained cnn model into memory.
    """
    try:
        infer.load_defog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
