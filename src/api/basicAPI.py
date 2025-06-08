from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
import mlflow
import joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = "data/processed/xgboost_model.pkl"
model = None

def get_best_model():
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Default")  # Or your experiment name
    if experiment is None:
        raise HTTPException(status_code=503, detail="No MLflow experiment found.")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])
    if not runs:
        raise HTTPException(status_code=503, detail="No runs found in MLflow.")
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def get_model():
    global model
    if model is None:
        model = get_best_model()
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