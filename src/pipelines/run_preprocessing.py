import os
from src.data.data_ingestion import ingest_data
from src.features.data_transformation import transform_data

def main():
    df = ingest_data()
    X_train, X_test, y_train, y_test = transform_data(df)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

if __name__ == "__main__":
    main()