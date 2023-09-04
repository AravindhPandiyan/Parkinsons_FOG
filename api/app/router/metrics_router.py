from typing import Union

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from api.app.controller import MetricsController
from api.app.dependencies import (
    APIResponseModel,
    ArchitectureTypes,
    ModelTypesModel,
    UsableData,
)
from api.app.models import MetricsResponseModel
from logger_config import logger as log

router = APIRouter()
controller = MetricsController()


@router.get("/", response_model=Union[APIResponseModel, MetricsResponseModel])
async def test_model(test: ModelTypesModel):
    """
    `test_model` is an API route for testing the different model's.

    Params:
        `test`: test is the data received from the user, containing request of a specific model to
        be **tested**.

    Returns:
        The return values of the function are dependent on the state of the API.
    """

    try:
        msg = None

        if test.use_data == UsableData.TDCSFOG:
            if test.architecture == ArchitectureTypes.RNN:
                msg = controller.test_tdcsfog_rnn()

            else:
                msg = controller.test_tdcsfog_cnn()

        elif test.use_data == UsableData.DEFOG:
            if test.architecture == ArchitectureTypes.RNN:
                msg = controller.test_defog_rnn()

            else:
                msg = controller.test_defog_cnn()

        if isinstance(msg, dict):
            return JSONResponse(status_code=status.HTTP_200_OK, content=msg)

        else:
            log.warning(msg)
            resp = {"detail": msg}
            return JSONResponse(
                status_code=status.HTTP_424_FAILED_DEPENDENCY, content=resp
            )

    except Exception as e:
        log.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The Server has a Boo Boo.",
        )
