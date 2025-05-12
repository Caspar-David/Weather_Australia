# tests/test_data_transformation.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.features.data_transformation import transform_data

def get_sample_from_data(path="data/weatherAUS.csv", n=10):
    df = pd.read_csv(path)
    df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})
    df = df.dropna(subset=["RainTomorrow"])
    return df.sample(n=n, random_state=42)

def test_transform_data_shapes():
    df = get_sample_from_data()
    X_train, X_test, y_train, y_test = transform_data(df)

    # Assert shapes match
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # PCA hat reduziert
    assert X_train.shape[1] < df.shape[1]