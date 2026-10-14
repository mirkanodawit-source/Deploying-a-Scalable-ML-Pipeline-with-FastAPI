"""Prediction endpoint uses the same preprocessing as training."""

from fastapi import APIRouter, HTTPException, Request

from app.schemas.prediction import CensusRecord, Prediction

router = APIRouter()


@router.post("/predict", response_model=Prediction)
def predict(record: CensusRecord, request: Request):
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return Prediction(prediction=service.predict(record))
