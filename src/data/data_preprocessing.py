import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def ingest_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.copy()
    df.dropna(subset=['RainTomorrow'], inplace=True)
    df.drop("Date", axis="columns", inplace=True)
    # Fill numerical columns
    num_cols = [
        "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
        "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
        "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
        "Temp9am", "Temp3pm"
    ]
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    # Fill categorical columns
    cat_cols = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    # One-hot encode
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    ohe_cols = enc.fit_transform(df[["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]])
    df = pd.concat([df, ohe_cols], axis=1)
    df.drop(["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"], axis=1, inplace=True)
    df['RainTomorrow'] = df['RainTomorrow'].replace({'No': 0, 'Yes': 1})
    return df

def split_and_transform(df):
    y = df['RainTomorrow']
    X = df.drop(['RainTomorrow'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train_res, y_test

def run_preprocessing_pipeline(path):
    df = ingest_data(path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_and_transform(df)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline("./data/raw/weatherAUS.csv")
    print("Preprocessing complete. Shapes:", X_train.shape, X_test.shape)