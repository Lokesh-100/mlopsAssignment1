from fastapi.testclient import TestClient
from app import main

client = TestClient(main.app)


def valid_payload():
    return {
        "age": 52,
        "sex": 1,
        "cp": 0,
        "trestbps": 125,
        "chol": 212,
        "fbs": 0,
        "restecg": 1,
        "thalach": 168,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }


def test_predict_success():
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    body = response.json()

    assert "prediction" in body
    assert "confidence" in body
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_missing_field():
    payload = valid_payload()
    payload.pop("age")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_type():
    payload = valid_payload()
    payload["age"] = "fifty"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
