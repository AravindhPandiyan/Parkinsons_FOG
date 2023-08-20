from fastapi import APIRouter, HTTPException

from api.app.controller import MetricsController

router = APIRouter()
controller = MetricsController()


@router.post('/tdcsfog/rnn', status_code=200)
def test_tdcsfog_rnn():
    try:
        controller.test_tdcsfog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/tdcsfog/cnn', status_code=200)
def test_tdcsfog_cnn():
    try:
        controller.test_tdcsfog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog/rnn', status_code=200)
def test_defog_rnn():
    try:
        controller.test_defog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog/cnn', status_code=200)
def test_defog_cnn():
    try:
        controller.test_defog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
