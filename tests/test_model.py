import joblib

def test_model_exists():
    model = joblib.load("models/heart_model.pkl")
    assert model is not None
