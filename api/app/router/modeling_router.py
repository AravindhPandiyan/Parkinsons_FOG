from fastapi import APIRouter, HTTPException

from api.app.controller import ModelingController

router = APIRouter()
modeler = ModelingController()


@router.post('/build/tdcsfog/rnn', status_code=201)
def build_tdcsfog_rnn():
    try:
        modeler.build_tdcsfog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/build/tdcsfog/cnn', status_code=201)
def build_tdcsfog_cnn():
    try:
        modeler.build_tdcsfog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/build/defog/rnn', status_code=201)
def build_defog_rnn():
    try:
        modeler.build_defog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/build/defog/cnn', status_code=201)
def build_defog_cnn():
    try:
        modeler.build_defog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/tdcsfog/rnn', status_code=200)
def train_tdcsfog_rnn():
    try:
        modeler.train_tdcsfog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/tdcsfog/cnn', status_code=200)
def train_tdcsfog_cnn():
    try:
        modeler.train_tdcsfog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/defog/rnn', status_code=200)
def train_defog_rnn():
    try:
        modeler.train_defog_rnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/train/defog/cnn', status_code=200)
def train_defog_cnn():
    try:
        modeler.train_defog_cnn()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
