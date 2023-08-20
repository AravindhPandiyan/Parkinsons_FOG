from fastapi import APIRouter, HTTPException

from api.app.controller import PreprocessorController

router = APIRouter()
controller = PreprocessorController()


@router.post('/tdcsfog/rnn', status_code=200)
def process_tdcsfog_rnn():
    try:
        controller.process_tdcsfog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/tdcsfog/cnn', status_code=200)
def process_tdcsfog_cnn():
    try:
        controller.process_tdcsfog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog/rnn', status_code=200)
def process_defog_rnn():
    try:
        controller.process_defog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog/cnn', status_code=200)
def process_defog_cnn():
    try:
        controller.process_defog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
