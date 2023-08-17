from fastapi import APIRouter, HTTPException

from api.app.controller import InferenceController

router = APIRouter()
infer = InferenceController()


@router.post('/tdcsfog', status_code=200)
def load_tdcsfog():
    try:
        infer.load_tdcsfog()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/defog', status_code=200)
def load_defog():
    try:
        infer.load_defog()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
