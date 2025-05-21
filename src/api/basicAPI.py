from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model at startup
model = joblib.load("data/processed/xgboost_model.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    # Convert input dict to DataFrame (expects correct feature names)
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}