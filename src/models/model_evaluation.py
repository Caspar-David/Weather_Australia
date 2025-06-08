from sklearn.metrics import accuracy_score

# This script evaluates a trained model using test data and returns the accuracy score.
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
