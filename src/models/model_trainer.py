from xgboost import XGBClassifier
import joblib
import yaml
import os

def load_params():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    return params["XGBoost"]

# This module trains an XGBoost model and saves it to a file.
def train_model(X_train, y_train):
    params = load_params()
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, "data/processed/xgboost_model.pkl")
    return model