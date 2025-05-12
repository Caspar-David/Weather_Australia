from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
