import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.model_loader import models

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["num_features"] == 78
    assert len(data["multiclass_classes"]) == 12


def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 400


def test_predict_accepts_valid_input():
    dummy_features = [0.0] * len(models.feature_columns)
    response = client.post("/predict", json={"features": dummy_features})
    assert response.status_code == 200
    data = response.json()
    assert "is_attack" in data
    assert "attack_probability" in data


def test_predict_missing_features_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # pydantic validation error
