from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from api.app.controller import PreprocessorController
from api.app.dependencies import APIResponseModel, ArchitectureTypes, ModelTypesModel, UsableData

router = APIRouter()
controller = PreprocessorController()


@router.post('/', response_model=APIResponseModel)
async def process(build: ModelTypesModel):
    try:
        if build.use_data == UsableData.TDCSFOG:
            if build.architecture == ArchitectureTypes.RNN:
                controller.process_tdcsfog_rnn()

            else:
                controller.process_tdcsfog_cnn()

        elif build.use_data == UsableData.DEFOG:
            if build.architecture == ArchitectureTypes.RNN:
                controller.process_defog_rnn()

            else:
                controller.process_defog_cnn()

        msg = f"""{build.use_data} for training the {build.architecture} model has been processed and converted to 
        TFRecords."""
        resp = {'detail': msg}
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=resp)

    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='The Server has a Boo Boo.')
