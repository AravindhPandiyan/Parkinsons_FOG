from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from api.app.controller import PreprocessorController
from api.app.dependencies import (
    APIResponseModel,
    ArchitectureTypes,
    ModelTypesModel,
    UsableData,
)
from logger_config import logger as log

router = APIRouter()
controller = PreprocessorController()


@router.post("/", response_model=APIResponseModel)
async def process(digest: ModelTypesModel):
    """
    `process` is an API route for **preprocessing** the different options of data in for each model architecture.

    Params:
        `digest`: digest is the data received from the user, containing the type of
        streaming requested.

    Returns:
        The return values of the function are dependent on the state of the API.
    """
    log.info("API Call")

    try:
        if digest.use_data == UsableData.TDCSFOG:
            if digest.architecture == ArchitectureTypes.RNN:
                controller.process_tdcsfog_rnn()

            else:
                controller.process_tdcsfog_cnn()

        elif digest.use_data == UsableData.DEFOG:
            if digest.architecture == ArchitectureTypes.RNN:
                controller.process_defog_rnn()

            else:
                controller.process_defog_cnn()

        msg = (
            f"{digest.use_data} for training the {digest.architecture} model has been processed and converted to "
            "TFRecords."
        )
        log.info(msg)
        resp = {"detail": msg}
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=resp)

    except Exception as e:
        log.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The Server has a Boo Boo.",
        )
