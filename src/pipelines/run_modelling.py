import pandas as pd
import mlflow
import mlflow.sklearn
from src.models.model_trainer import train_model
from src.models.model_evaluation import evaluate_model

# This script trains an XGBoost model using processed data and logs the model and metrics to MLflow.
def main():
    # Set MLflow tracking URI to the mlflow service
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
    )

    best_acc = runs[0].data.metrics.get("accuracy", 0.0) if runs else 0.0  # fallback if no runs yet
    
    
    # Start MLflow run
    with mlflow.start_run() as run: # as run is added
        # Train model
        model = train_model(X_train, y_train)
        run_name = mlflow.get_run(run.info.run_id).data.tags.get('mlflow.runName', run.info.run_id) # this line is added
        # Evaluate model
        acc = evaluate_model(model, X_test, y_test)

        # Log model and metrics
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", acc)

        # Save model to processed directory
        import joblib
        if acc > best_acc:
            joblib.dump(model, "data/processed/xgboost_model.pkl")
            with open("data/processed/model_run_name.txt", "w") as f:  # these 2 lines are added
                f.write(f"{run_name}: acc={acc:.5f}")#(run_name)


'''
def main():
    # Set MLflow tracking URI to the mlflow service
    
    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
        
    # Start MLflow run
    
        # Train model
    model = train_model(X_train, y_train)
        
    acc = evaluate_model(model, X_test, y_test)
    print("ACC: ",acc)
'''

if __name__ == "__main__":
    main()