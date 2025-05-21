import pandas as pd
from src.models.model_trainer import train_model
from src.models.model_evaluation import evaluate_model

def main():
    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model to processed directory
    import joblib
    joblib.dump(model, "data/processed/xgboost_model.pkl")

if __name__ == "__main__":
    main()