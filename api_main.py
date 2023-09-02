"""
Main FastAPI Application
------------------------

This module contains the main FastAPI application that serves various API routes for different functionalities.
It includes routers for inference, metrics, modeling, and preprocessing.

Usage:

- Run this script directly to start the FastAPI application using Uvicorn.

- The API will be available at http://0.0.0.0:8080.

Example:

    To start the server, run the below command from the root folder of the project:
    ```
    uvicorn api.main:app --host localhost --port 8080 --reload
    ```

Modules:

- `uvicorn`: ASGI server to run the FastAPI application.

- `fastapi.FastAPI`: The FastAPI framework for building APIs.

Routes:

- `/inference`: API route for inference-related operations.

- `/metrics`: API route for metrics related operations.

- `/modeling`: API route for modeling-related operations.

- `/preprocessing`: API route for preprocessing related operations.

"""

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
    uvicorn.run("api_main:app", host="localhost", port=8080, reload=True)
