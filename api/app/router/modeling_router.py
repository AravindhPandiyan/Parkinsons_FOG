import concurrent.futures

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse

from api.app.controller import ModelingController
from api.app.dependencies import (
    APIResponseModel,
    ArchitectureTypes,
    ModelTypesModel,
    UsableData,
)
from logger_config import logger as log

router = APIRouter()
controller = ModelingController()
executor = concurrent.futures.ThreadPoolExecutor()
background_task_status = {}


def _background_training(task_data: ModelTypesModel):
    """
    `_background_streaming` is a private funtion used for setting up **training** of model as a background task in
    an API call.

    Params:
        `task_data`: task_data is used for beginning the training of a specific model mentioned
        in it.
    """
    log.info("Funtion Call")

    if task_data.use_data == UsableData.TDCSFOG:
        if task_data.architecture == ArchitectureTypes.RNN:
            controller.train_tdcsfog_rnn()

        else:
            controller.train_tdcsfog_cnn()

    elif task_data.use_data == UsableData.DEFOG:
        if task_data.architecture == ArchitectureTypes.RNN:
            controller.train_defog_rnn()

        else:
            controller.train_defog_cnn()

    result = f"x{task_data.architecture} model training using {task_data.use_data} data has been completed."
    background_task_status[
        f"{task_data.use_data}_{task_data.architecture}"
    ] = "completed"
    log.info(result)
    print(f"\n{result}")


@router.post("/build", response_model=APIResponseModel)
async def build_model(build: ModelTypesModel):
    """
    `build_model` is an API route for **building** the different model's.

    Params:
        `build`: build is the data received from the user, containing the model requested by
        the user to be **constructed**.

    Returns:
        The return values of the function are dependent on the state of the API.

    """
    log.info("API Call")

    try:
        if build.use_data == UsableData.TDCSFOG:
            if build.architecture == ArchitectureTypes.RNN:
                controller.build_tdcsfog_rnn()

            else:
                controller.build_tdcsfog_cnn()

        elif build.use_data == UsableData.DEFOG:
            if build.architecture == ArchitectureTypes.RNN:
                controller.build_defog_rnn()

            else:
                controller.build_defog_cnn()

        msg = f"{build.architecture} model has been constructed to train on {build.use_data} data."
        resp = {"detail": msg}
        log.info(msg)
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=resp)

    except Exception as e:
        log.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The Server has a Boo Boo.",
        )


@router.post("/train", response_model=APIResponseModel)
async def train_model(train: ModelTypesModel, background_tasks: BackgroundTasks):
    """
    `train_model` is an API route for the **training** of the different model's.

    Params:
        `train`: train is the data received from the user, containing request of a specific
        model requested to be trained.

        `background_tasks`: background_tasks is the parameter passed to the funtion by the
        decorator funtion, It is used for running any long-running task or API-freezing task to run in the background.

    Returns:
        The return values of the function are dependent on the state of the API.
    """
    log.info("API Call")

    if (
        background_task_status.get(f"{train.use_data}_{train.architecture}")
        == "running"
    ):
        msg = f"Training of {train.architecture} model using {train.use_data} data is already running."
        log.warning(msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    elif controller.MODEL_TYPE != f"{train.use_data}_{train.architecture}":
        msg = f"Please First Build the {train.use_data} {train.architecture} model to train it."
        log.warning(msg)
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=msg)

    else:
        try:
            background_task_status[f"{train.use_data}_{train.architecture}"] = "running"
            background_tasks.add_task(_background_training, train)
            msg = (
                f"Training of {train.architecture} model using {train.use_data} data has been enqueued for "
                "background execution."
            )
            log.warning(msg)
            resp = {"detail": msg}
            return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

        except Exception as e:
            log.error(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The Server has a Boo Boo.",
            )
