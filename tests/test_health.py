from fastapi.testclient import TestClient
import pytest

from app.main import create_app


def test_greeting_and_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Hello" in response.json()["message"]
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_missing_model_fails_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path)):
            pass


def test_not_ready_returns_503():
    client = TestClient(create_app())
    assert client.get("/ready").status_code == 503
