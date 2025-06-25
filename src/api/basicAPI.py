from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from prometheus_client import Gauge
from fastapi.responses import Response
import secrets
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import time
import os

app = FastAPI()

# --- Basic Auth Setup ---
security = HTTPBasic()

VALID_USERS = {
    "airflow": "airflow",
    "admin": "admin",
    "user": "user"
}

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    if username not in VALID_USERS or not secrets.compare_digest(password, VALID_USERS[username]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return username

# --- Prometheus Metrics ---
API_REQUESTS = Counter('api_requests_total', 'Total API calls', ['endpoint', 'status'])
API_LATENCY = Histogram('api_latency_seconds', 'API latency', ['endpoint'])
MODEL_LOAD_TIME = Histogram('model_load_seconds', 'Time to load model')
model_info_requests = Counter("model_info_requests_total", "Total requests to /model-info endpoint")
PREDICT_REQUESTS_TOTAL = Counter("predict_requests_total", "Total number of times the /predict endpoint was called")
model_accuracy = Gauge("model_accuracy", "Accuracy of the best ML model")
model_run_name = Gauge("model_run_name", "Run name of the best ML model", ['run_name'])

# --- MLflow & Model Setup ---
MODEL_PATH = "data/processed/xgboost_model.pkl"
RUN_NAME_PATH = "data/processed/model_run_name.txt"

model = None
model_time = None

def get_best_model():
    start_time = time.time()
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    if experiment is None:
        raise HTTPException(status_code=503, detail="No MLflow experiment found.")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])
    if not runs:
        raise HTTPException(status_code=503, detail="No runs found in MLflow.")
    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    MODEL_LOAD_TIME.observe(time.time() - start_time)
    return loaded_model

def get_model():
    global model, model_time
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not available yet.")
    checked_mtime = os.path.getmtime(MODEL_PATH)
    if model is None or model_time != checked_mtime:
        # Here we rely on MLflow loading, or joblib if you prefer
        model = get_best_model()
        model_time = checked_mtime
    return model

# --- Endpoints ---

@app.get("/health")
async def health(user: str = Depends(authenticate)):
    API_REQUESTS.labels(endpoint="health", status="success").inc()
    return {"status": "ok"}

@app.post("/predict")
async def predict(features: dict, user: str = Depends(authenticate)):
    start_time = time.time()
    try:
        model = get_model()
        X = pd.DataFrame([features])
        prediction = model.predict(X)[0]
        API_REQUESTS.labels(endpoint="predict", status="success").inc()
        PREDICT_REQUESTS_TOTAL.inc()
        API_LATENCY.labels(endpoint="predict").observe(time.time() - start_time)
        return {"prediction": int(prediction)}
    except Exception as e:
        API_REQUESTS.labels(endpoint="predict", status="error").inc()
        API_LATENCY.labels(endpoint="predict").observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info(user: str = Depends(authenticate)):
    model_info_requests.inc()  # Prometheus counter
    
    if not os.path.exists(RUN_NAME_PATH):
        raise HTTPException(status_code=503, detail="Run name file not available yet.")
    
    with open(RUN_NAME_PATH) as f:
        content = f.read().strip()

    if not content:
        raise HTTPException(status_code=503, detail="Run name file is empty.")  

    if ": acc=" in content:
        name, acc = content.split(": acc=")
        acc = float(acc)

        # Prometheus Metrics
        model_accuracy.set(acc)
        model_run_name.labels(run_name=name).set(1)

        return {"run_name": name, "accuracy": acc}

@app.get("/metrics")
async def metrics(user: str = Depends(authenticate)):
    return Response(generate_latest(REGISTRY), media_type="text/plain")

@app.post("/update-metrics")
def update_metrics(run_name: str):
    model_run_name.labels(run_name=run_name).set(1)
    return {"status": "updated"}