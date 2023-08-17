from fastapi import APIRouter, HTTPException

from api.app.controller import PreprocessorController

router = APIRouter()
controller = PreprocessorController()


@router.post('/tdcsfog', status_code=200)
def process_tdcsfog():
    try:
        controller.process_tdcsfog()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog', status_code=200)
def process_defog():
    try:
        controller.process_defog()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
