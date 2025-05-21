import pandas as pd

def ingest_data(filepath="./data/raw/weatherAUS.csv"):
    df = pd.read_csv(filepath)
    df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})
    df = df.dropna(subset=["RainTomorrow"])
    return df