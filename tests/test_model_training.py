# tests/test_model_training.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model_trainer import train_model
import pandas as pd

def test_train_model_runs():
    df = pd.read_csv("tests/sample_weather.csv")
    df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["RainTomorrow"])

    from src.features.data_transformation import transform_data
    X_train, _, y_train, _ = transform_data(df)

    model = train_model(X_train, y_train)
    assert hasattr(model, "predict")