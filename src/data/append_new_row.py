import pandas as pd
import os

RAW_PATH = "data/raw/weatherAUS.csv"
FUTURE_PATH = "data/raw/weatherAUS_future.csv"

def main():
    print("=== APPEND SCRIPT STARTED ===")
    print("Current working directory:", os.getcwd())
    print("Listing /app/data/raw:", os.listdir("data/raw"))
    print("RAW_PATH exists:", os.path.exists(RAW_PATH))
    print("FUTURE_PATH exists:", os.path.exists(FUTURE_PATH))
    print("RAW_PATH absolute:", os.path.abspath(RAW_PATH))
    print("FUTURE_PATH absolute:", os.path.abspath(FUTURE_PATH))

    if not os.path.exists(FUTURE_PATH):
        print("No future data left to append.")
        return

    df_raw = pd.read_csv(RAW_PATH)
    df_future = pd.read_csv(FUTURE_PATH)

    print(f"Rows in RAW before append: {len(df_raw)}")
    print(f"Rows in FUTURE before append: {len(df_future)}")

    if df_future.empty:
        print("No more rows to append.")
        return

    # Take the first row from future and append to raw
    new_row = df_future.iloc[[0]]
    df_raw = pd.concat([df_raw, new_row], ignore_index=True)
    df_future = df_future.iloc[1:]

    df_raw.to_csv(RAW_PATH, index=False)
    df_future.to_csv(FUTURE_PATH, index=False)

    print(f"Rows in RAW after append: {len(df_raw)}")
    print(f"Rows in FUTURE after append: {len(df_future)}")
    print("Appended one new row to raw data.")

if __name__ == "__main__":
    main()