from xgboost import XGBClassifier
import joblib

def train_model(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "xgboost_model.pkl")
    return model
