import uvicorn
from fastapi import FastAPI

from api.app.router import (
    inference_router,
    metrics_router,
    modeling_router,
    preprocessing_router,
)

app = FastAPI()

app.include_router(inference_router.router, prefix="/inference")
app.include_router(metrics_router.router, prefix="/metrics")
app.include_router(modeling_router.router, prefix="/modeling")
app.include_router(preprocessing_router.router, prefix="/preprocessing")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
