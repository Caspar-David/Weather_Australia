import pandas as pd
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

with open("schema.yaml", "r") as file:
    schema = yaml.safe_load(file)

def validate_data():
    data_path = config["data_validation"]["unzip_dir"]
    status_file = config["data_validation"]["STATUS_FILE"]

    df = pd.read_csv(data_path)

    for column, dtype in schema["COLUMNS"].items():
        if column not in df.columns:
            print(f"Missing column: {column}")
            return False
        if str(df[column].dtype) != dtype:
            print(f"Incorrect dtype for {column}. Expected {dtype}, got {df[column].dtype}")
            return False

    with open(status_file, "w") as f:
        f.write("Validation Passed")

    print("Data validation passed.")
    return True

if __name__ == "__main__":
    validate_data()

