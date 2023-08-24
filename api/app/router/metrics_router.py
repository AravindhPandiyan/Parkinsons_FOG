from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from api.app.controller import MetricsController

router = APIRouter()
controller = MetricsController()


@router.get('/tdcsfog/rnn', status_code=status.HTTP_200_OK)
async def test_tdcsfog_rnn():
    """
    test_tdcsfog_rnn is api router path for testing the tdcsfog data trained rnn model.
    """
    try:
        msg = controller.test_tdcsfog_rnn()

        if isinstance(msg, dict):
            return msg
        else:
            return JSONResponse(status_code=status.HTTP_424_FAILED_DEPENDENCY, content={"details": msg})

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The Server has a Boo Boo.')


@router.get('/tdcsfog/cnn', status_code=status.HTTP_200_OK)
async def test_tdcsfog_cnn():
    """
    test_tdcsfog_cnn is api router path for testing the tdcsfog data trained cnn model.
    """
    try:
        msg = controller.test_tdcsfog_cnn()

        if isinstance(msg, dict):
            return msg
        else:
            return JSONResponse(status_code=status.HTTP_424_FAILED_DEPENDENCY, content={"details": msg})

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The Server has a Boo Boo.')


@router.get('/defog/rnn', status_code=status.HTTP_200_OK)
async def test_defog_rnn():
    """
    test_defog_rnn is api router path for testing the defog data trained rnn model.
    """
    try:
        msg = controller.test_defog_rnn()

        if isinstance(msg, dict):
            return msg
        else:
            return JSONResponse(status_code=status.HTTP_424_FAILED_DEPENDENCY, content={"details": msg})

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The Server has a Boo Boo.')


@router.get('/defog/cnn', status_code=status.HTTP_200_OK)
async def test_defog_cnn():
    """
    test_defog_cnn is api router path for testing the defog data trained cnn model.
    """
    try:
        msg = controller.test_defog_cnn()

        if isinstance(msg, dict):
            return msg
        else:
            return JSONResponse(status_code=status.HTTP_424_FAILED_DEPENDENCY, content={"details": msg})

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The Server has a Boo Boo.')
