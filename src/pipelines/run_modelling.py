import pandas as pd
import mlflow
import mlflow.sklearn
from src.models.model_trainer import train_model
from src.models.model_evaluation import evaluate_model
import os

def main():
    # Set MLflow tracking URI to the mlflow service
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        acc = evaluate_model(model, X_test, y_test)

        # Log model and metrics
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", acc)

        # Save model to processed directory
        import joblib
        joblib.dump(model, "data/processed/xgboost_model.pkl")

if __name__ == "__main__":
    main()