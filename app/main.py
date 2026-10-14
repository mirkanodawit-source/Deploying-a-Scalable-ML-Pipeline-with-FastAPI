"""FastAPI application factory and compatibility entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import health, predict
from app.core.config import MODEL_DIR
from app.services.model_service import ModelService


def create_app(model_dir=None):
    artifact_dir = Path(model_dir) if model_dir is not None else MODEL_DIR

    @asynccontextmanager
    async def lifespan(application):
        # Fail startup rather than silently serve a dummy model.
        application.state.model_service = ModelService(artifact_dir)
        yield
        application.state.model_service = None

    application = FastAPI(
        title="Census Income Classifier", version="1.0.0", lifespan=lifespan,
        description="Educational classification of Census income above $50K.",
    )
    application.include_router(health.router)
    application.include_router(predict.router)
    return application


app = create_app()
