"""Greeting and health routes."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/")
def welcome():
    return {"message": "Hello from the Census income prediction API!"}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request):
    if getattr(request.app.state, "model_service", None) is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ready"}
