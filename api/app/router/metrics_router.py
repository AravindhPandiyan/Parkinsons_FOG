from fastapi import APIRouter, HTTPException

from api.app.controller import MetricsController

router = APIRouter()
controller = MetricsController()


@router.get('/tdcsfog/rnn', status_code=200)
async def test_tdcsfog_rnn():
    """
    test_tdcsfog_rnn is api router path for testing the tdcsfog data trained rnn model.
    """
    try:
        controller.test_tdcsfog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/tdcsfog/cnn', status_code=200)
async def test_tdcsfog_cnn():
    """
    test_tdcsfog_cnn is api router path for testing the tdcsfog data trained cnn model.
    """
    try:
        return controller.test_tdcsfog_cnn()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/defog/rnn', status_code=200)
async def test_defog_rnn():
    """
    test_defog_rnn is api router path for testing the defog data trained rnn model.
    """
    try:
        return controller.test_defog_rnn()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/defog/cnn', status_code=200)
async def test_defog_cnn():
    """
    test_defog_cnn is api router path for testing the defog data trained cnn model.
    """
    try:
        return controller.test_defog_cnn()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
