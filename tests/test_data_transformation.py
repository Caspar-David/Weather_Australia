# tests/test_data_transformation.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.features.data_transformation import transform_data

def get_sample_from_data(path="data/raw/weatherAUS.csv", n=100):
    df = pd.read_csv(path)
    df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)
    df = df.dropna(subset=["RainTomorrow"])
    return df.sample(n=n, random_state=42)
