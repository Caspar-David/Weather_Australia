import pandas as pd
import mlflow
import mlflow.sklearn
from src.models.model_trainer import train_model
from src.models.model_evaluation import evaluate_model
import requests
import joblib

# This script trains an XGBoost model using processed data and logs the model and metrics to MLflow.
def main():
    # Set MLflow tracking URI to the mlflow service
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Start MLflow run
    with mlflow.start_run() as run:
        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        acc = evaluate_model(model, X_test, y_test)

        # Log model and metrics
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", acc)

        # Save model to processed directory
        
        joblib.dump(model, "/app/data/processed/xgboost_model.pkl")

        # Save run name and accuracy to file for /model-info API
        run_name = run.info.run_name or f"run_{run.info.run_id}"
        output_path = "/app/data/processed/model_run_name.txt"

        with open(output_path, "w") as f:
            f.write(f"{run_name}: acc={acc:.4f}")

        try:
            response = requests.post(
                "http://api:8000/update-metrics",
                json={"run_name": run_name}
            )
            print("Prometheus Metric updated:", response.json())
        except Exception as e:
            print("Error updating metric:", e)

if __name__ == "__main__":
    main()