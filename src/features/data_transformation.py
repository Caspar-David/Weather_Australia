import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def transform_data(df):
    X = df.drop(columns=["RainTomorrow"])
    y = df["RainTomorrow"]

    X = pd.get_dummies(X)  # basic encoding, improve later if needed

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
