from src.data.data_ingestion import ingest_data
from src.features.data_transformation import transform_data
from src.models.model_trainer import train_model
from src.models.model_evaluation import evaluate_model

def run_pipeline():
    df = ingest_data()
    X_train, X_test, y_train, y_test = transform_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()

