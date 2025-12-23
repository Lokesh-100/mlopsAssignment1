import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("models/heart_model.pkl")

class Patient(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: Patient):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[0][1]
    pred = int(prob > 0.5)
    return {
        "prediction": pred,
        "confidence": prob
    }
