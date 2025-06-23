from fastapi import FastAPI, HTTPException , Depends,  status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import joblib
import pandas as pd
import os

from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI()
security = HTTPBasic()

# === Prometheus Counter ===
model_info_requests = Counter("model_info_requests_total", "Total requests to /model-info endpoint")

predict_requests_total = Counter(
    "predict_requests_total",
    "Total number of times the /predict endpoint was called"
)

MODEL_PATH = "data/processed/xgboost_model.pkl"
RUN_NAME_PATH = "data/processed/model_run_name.txt" # added
model = None
run_name = None # is added
model_time = None #  modification time is added

VALID_USERS = {
    "admin": "admin",
    "user": "user"
}

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password

    if username not in VALID_USERS:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    expected_password = VALID_USERS[username]
    if not secrets.compare_digest(password, expected_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return username

'''
def get_model():
    global model 
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model not available yet.")
        model = joblib.load(MODEL_PATH)
    return model
'''
def get_model():
    global model, model_time

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not available yet.")

    checked_mtime = os.path.getmtime(MODEL_PATH)# return the time of saving the model

    if model is None or model_time != checked_mtime:
        model = joblib.load(MODEL_PATH)
        model_time = checked_mtime  # Update stored modification time

    return model




@app.get("/health")
def health(user: str = Depends(authenticate)):
    return {"status": "ok"}

@app.post("/predict")
def predict(features: dict):
    predict_requests_total.inc()
    model = get_model()
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}

#### added ####
@app.get("/model-info")
def model_info(user: str = Depends(authenticate)):
    #global run_name
    #if run_name is None:
        model_info_requests.inc()  # Increment Prometheus counter
        
        if not os.path.exists(RUN_NAME_PATH):
            raise HTTPException(status_code=503, detail="Run name file not available yet.")
        with open(RUN_NAME_PATH) as f:
            #run_name = f.read().strip()
            content = f.read().strip()
        if not content:
            raise HTTPException(status_code=503, detail="Run name file is empty.")  
        
        if ": acc=" in content:
            name, acc = content.split(": acc=")
            return {"run_name": name, "accuracy": float(acc)}  
        #return {"run_name": run_name}
        
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)        