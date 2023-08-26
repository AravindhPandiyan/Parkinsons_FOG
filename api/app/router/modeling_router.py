import concurrent.futures

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import JSONResponse

from api.app.controller import ModelingController
from api.app.dependencies import APIResponseModel, ModelTypesModel, UsableData, ArchitectureTypes

router = APIRouter()
controller = ModelingController()
executor = concurrent.futures.ThreadPoolExecutor()
background_task_status = {}


def background_training(task_data):
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

    result = f'\n{task_data.architecture} model training using {task_data.use_data} data has been completed.'
    background_task_status[f'{task_data.use_data}_{task_data.architecture}'] = "completed"
    print(result)


@router.post('/build', response_model=APIResponseModel)
async def build_model(build: ModelTypesModel):
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

        msg = f'{build.architecture} model has been constructed to train on {build.use_data} data.'
        resp = {'detail': msg}
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=resp)

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The Server has a Boo Boo.')


@router.post('/train', response_model=APIResponseModel)
async def train_model(train: ModelTypesModel, background_tasks: BackgroundTasks):
    if background_task_status.get(f'{train.use_data}_{train.architecture}') == 'running':
        msg = f"Training of {train.architecture} model using {train.use_data} data is already running."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    elif controller.MODEL_TYPE != f'{train.use_data}_{train.architecture}':
        msg = f'Please First Build the {train.use_data} {train.architecture} model to train it.'
        raise HTTPException(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=msg)

    else:
        try:
            background_task_status[f'{train.use_data}_{train.architecture}'] = "running"
            background_tasks.add_task(background_training, train)
            msg = f"Training of {train.architecture} model using {train.use_data} data has been enqueued for " \
                  "background execution."
            resp = {'detail': msg}
            return JSONResponse(status_code=status.HTTP_200_OK, content=resp)

        except Exception:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The Server has a Boo Boo.')
