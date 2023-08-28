Module api_main
===============
Main FastAPI Application
------------------------

This module contains the main FastAPI application that serves various API routes for different functionalities.
It includes routers for inference, metrics, modeling, and preprocessing.

Usage:

- Run this script directly to start the FastAPI application using Uvicorn.

- The API will be available at http://0.0.0.0:8000.

Example:

    To start the server:
    ```
    python api_main.py
    ```

Modules:

- `uvicorn`: ASGI server to run the FastAPI application.

- `fastapi.FastAPI`: The FastAPI framework for building APIs.

Routes:

- `/inference`: API route for inference-related operations.

- `/metrics`: API route for metrics related operations.

- `/modeling`: API route for modeling-related operations.

- `/preprocessing`: API route for preprocessing related operations.