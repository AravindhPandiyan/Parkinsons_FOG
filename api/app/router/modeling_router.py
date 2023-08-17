from fastapi import APIRouter, HTTPException

from api.app.controller import ModelingController

router = APIRouter()
modeler = ModelingController()


@router.post('/build/tdcsfog', status_code=201)
def build_tdcsfog():
    try:
        modeler.build_tdcsfog()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/build/defog', status_code=201)
def build_defog():
    try:
        modeler.build_defog()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/tdcsfog', status_code=200)
def train_tdcsfog():
    try:
        modeler.train_tdcsfog()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/defog', status_code=200)
def train_defog():
    try:
        modeler.train_defog()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
