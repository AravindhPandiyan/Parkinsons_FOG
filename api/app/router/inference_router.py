from fastapi import APIRouter, HTTPException

from api.app.controller import InferenceController

router = APIRouter()
infer = InferenceController()


@router.post('/tdcsfog/rnn', status_code=200)
def load_tdcsfog_rnn():
    try:
        infer.load_tdcsfog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/tdcsfog/cnn', status_code=200)
def load_tdcsfog_cnn():
    try:
        infer.load_tdcsfog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog/rnn', status_code=200)
def load_defog_rnn():
    try:
        infer.load_defog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog/cnn', status_code=200)
def load_defog_cnn():
    try:
        infer.load_defog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
