from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = "data/processed/xgboost_model.pkl"
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model not available yet.")
        model = joblib.load(MODEL_PATH)
    return model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    model = get_model()
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}