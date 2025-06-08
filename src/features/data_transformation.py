import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

# This module transforms the weather data for modeling by handling missing values, encoding categorical variables, and balancing the dataset.
def transform_data(df):
    # Handle missing values for the target attribute and drop date column
    df.dropna(subset=['RainTomorrow'], inplace=True)
    df.drop("Date", axis="columns", inplace=True)

    # Definition numeric und categorical columns
    num_cols = [
        "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
        "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
        "Humidity9am", "Humidity3pm",
        "Pressure9am", "Pressure3pm",
        "Cloud9am", "Cloud3pm",
        "Temp9am", "Temp3pm"
    ]

    cat_cols = [
        "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"
    ]

    # Imputation: Num Cols
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Imputation: Cat Cols
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # One-Hot Encoding: Cat Cols
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    ohe_cols = enc.fit_transform(df[["Location"] + cat_cols])
    
    df = pd.concat([df, ohe_cols], axis=1)
    df.drop(["Location"] + cat_cols, axis=1, inplace=True)

    # Target and Dataset
    y = df["RainTomorrow"]
    X = df.drop(columns=["RainTomorrow"])

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE
    smote = SMOTE(random_state=0)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Return
    return X_train_res, X_test, y_train_res, y_test