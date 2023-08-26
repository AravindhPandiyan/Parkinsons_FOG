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

router = APIRouter()
controller = ModelingController()
executor = concurrent.futures.ThreadPoolExecutor()
background_task_status = {}


def _background_training(task_data):
    """
    _background_streaming is a private funtion used for setting up training of model ass a background task in an
    API call.
    :param task_data: task_data is used for beginning the training of a specific model mentioned in it.
    """
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

    result = f"\n{task_data.architecture} model training using {task_data.use_data} data has been completed."
    background_task_status[
        f"{task_data.use_data}_{task_data.architecture}"
    ] = "completed"
    print(result)


@router.post("/build", response_model=APIResponseModel)
async def build_model(build: ModelTypesModel):
    """
    build_model is an API route for building different model's.
    :param build: build is the data received from the user, containing the model requested by the user to be built.
    :return: The return values of the function is dependent on the state of API.
    """
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
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=resp)

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The Server has a Boo Boo.",
        )


@router.post("/train", response_model=APIResponseModel)
async def train_model(train: ModelTypesModel, background_tasks: BackgroundTasks):
    """
    train_model is an API route for the training of different types model's.
    :param train: train is the data received from the user, containing request of a specific model requested to be
    trained.
    :param background_tasks: background_tasks is the parameter passed to the funtion by the decorator funtion, It is
    used for running any long-running task or API freezing task to run in the background.
    :return: The return values of the function is dependent on the state of API.
    """
    if (
        background_task_status.get(f"{train.use_data}_{train.architecture}")
        == "running"
    ):
        msg = f"Training of {train.architecture} model using {train.use_data} data is already running."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    elif controller.MODEL_TYPE != f"{train.use_data}_{train.architecture}":
        msg = f"Please First Build the {train.use_data} {train.architecture} model to train it."
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=msg)

    else:
        try:
            background_task_status[f"{train.use_data}_{train.architecture}"] = "running"
            background_tasks.add_task(_background_training, train)
            msg = (
                f"Training of {train.architecture} model using {train.use_data} data has been enqueued for "
                "background execution."
            )
            resp = {"detail": msg}
            return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The Server has a Boo Boo.",
            )
