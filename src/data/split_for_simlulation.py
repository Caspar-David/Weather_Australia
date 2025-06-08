import pandas as pd
from sklearn.model_selection import train_test_split
import os

RAW_PATH = "data/raw/weatherAUS.csv"
INITIAL_PATH = "data/raw/weatherAUS_initial.csv"
FUTURE_PATH = "data/raw/weatherAUS_future.csv"

def main():
    df = pd.read_csv(RAW_PATH)
    df_initial, df_future = train_test_split(df, test_size=0.1, random_state=42)
    df_initial.to_csv(INITIAL_PATH, index=False)
    df_initial.to_csv(RAW_PATH, index=False)  # Overwrite with 90%
    df_future.to_csv(FUTURE_PATH, index=False)
    print(f"Initial (90%) rows: {len(df_initial)}, Future (10%) rows: {len(df_future)}")

if __name__ == "__main__":
    main()