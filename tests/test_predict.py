import pandas as pd
import pytest

from app.schemas.prediction import EXAMPLE
from ml.data import process_data
from ml.model import inference


def test_prediction_matches_direct_inference(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    service = client.app.state.model_service
    X, _, _, _ = process_data(
        pd.DataFrame([EXAMPLE]), training=False, encoder=service.encoder,
    )
    expected = service.lb.inverse_transform(inference(service.model, X))[0]
    assert response.json() == {"prediction": expected}


def test_predictions_cover_both_classes(client):
    low = {**EXAMPLE, "age": 18, "education": "11th", "education-num": 7,
           "capital-gain": 0, "hours-per-week": 10}
    high = {**EXAMPLE, "age": 50, "marital-status": "Married-civ-spouse",
            "relationship": "Husband", "capital-gain": 99999}
    assert client.post("/predict", json=low).json()["prediction"] == "<=50K"
    assert client.post("/predict", json=high).json()["prediction"] == ">50K"


@pytest.mark.parametrize("change", [
    {"age": -1}, {"age": "39"}, {"hours-per-week": 0},
    {"education-num": 17}, {"workclass": " "}, {"unexpected": True},
])
def test_invalid_input_rejected(client, change):
    assert client.post("/predict", json={**EXAMPLE, **change}).status_code == 422


def test_missing_feature_rejected(client):
    record = dict(EXAMPLE)
    record.pop("age")
    assert client.post("/predict", json=record).status_code == 422


def test_unknown_category_and_reordered_fields(client):
    record = {**EXAMPLE, "occupation": "Unseen-occupation"}
    response = client.post("/predict", json=dict(reversed(list(record.items()))))
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_openapi_documents_aliases(client):
    schema = client.get("/openapi.json").json()
    properties = schema["components"]["schemas"]["CensusRecord"]["properties"]
    assert "education-num" in properties
    assert "education_num" not in properties
