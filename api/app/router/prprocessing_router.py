from fastapi import APIRouter, HTTPException

from api.app.controller import PreprocessorController

router = APIRouter()
controller = PreprocessorController()


@router.post('/tdcsfog/rnn', status_code=200)
async def process_tdcsfog_rnn():
    """
    process_tdcsfog_rnn is api router path for processing the tdcsfog data for rnn model.
    """
    try:
        controller.process_tdcsfog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/tdcsfog/cnn', status_code=200)
async def process_tdcsfog_cnn():
    """
    process_tdcsfog_cnn is api router path for processing the tdcsfog data for cnn model.
    """
    try:
        controller.process_tdcsfog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog/rnn', status_code=200)
async def process_defog_rnn():
    """
    process_defog_rnn is api router path for processing the defog data for rnn model.
    """
    try:
        controller.process_defog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog/cnn', status_code=200)
async def process_defog_cnn():
    """
    process_defog_rnn is api router path for processing the defog data for cnn model.
    """
    try:
        controller.process_defog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
