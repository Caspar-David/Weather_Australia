import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import multiprocessing

# Load dataset
df = pd.read_csv("weatherAUS.csv")

# Check balance of the target attribute
df['RainTomorrow'].value_counts().plot(kind="bar", title="Rain Tomorrow Distribution Before SMOTE", rot=0)

# Handle missing values for the target attribute and drop date column
print(f"Number of missing values for target attribute: {df['RainTomorrow'].isna().sum()}")
df.dropna(subset=['RainTomorrow'], inplace=True)
df.drop("Date", axis="columns", inplace=True)

# Data Imputation
df.fillna({"MinTemp": df["MinTemp"].median()}, inplace=True)
df.fillna({"MaxTemp": df["MaxTemp"].median()}, inplace=True)
df.fillna({"Rainfall": df["Rainfall"].median()}, inplace=True)
df.fillna({"Evaporation": df["Evaporation"].median()}, inplace=True)
df.fillna({"Sunshine": df["Sunshine"].median()}, inplace=True)
df.fillna({"WindGustSpeed": df["WindGustSpeed"].median()}, inplace=True)
df.fillna({"WindSpeed9am": df["WindSpeed9am"].median()}, inplace=True)
df.fillna({"WindSpeed3pm": df["WindSpeed3pm"].median()}, inplace=True)
df.fillna({"Humidity9am": df["Humidity9am"].median()}, inplace=True)
df.fillna({"Humidity3pm": df["Humidity3pm"].median()}, inplace=True)
df.fillna({"Pressure9am": df["Pressure9am"].median()}, inplace=True)
df.fillna({"Pressure3pm": df["Pressure3pm"].median()}, inplace=True)
df.fillna({"Cloud9am": df["Cloud9am"].median()}, inplace=True)
df.fillna({"Cloud3pm": df["Cloud3pm"].median()}, inplace=True)
df.fillna({"Temp9am": df["Temp9am"].median()}, inplace=True)
df.fillna({"Temp3pm": df["Temp3pm"].median()}, inplace=True)

# Replace NaNs with Mode for categorical features
df.fillna({"WindGustDir": df["WindGustDir"].mode()[0]}, inplace=True)
df.fillna({"WindDir9am": df["WindDir9am"].mode()[0]}, inplace=True)
df.fillna({"WindDir3pm": df["WindDir3pm"].mode()[0]}, inplace=True)
df.fillna({"RainToday": df["RainToday"].mode()[0]}, inplace=True)

# One-Hot Encoding of categorical features
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
ohe_cols = enc.fit_transform(df[["Location", "WindGustDir", "WindDir9am", "WindDir3pm", 'RainToday']])

# Add One-Hot Encoded columns and drop redundant columns
df = pd.concat([df, ohe_cols], axis=1)
df = df.drop(["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"], axis=1)

# Replace target variable 'RainTomorrow' with binary labels (0 and 1)
df['RainTomorrow'] = df['RainTomorrow'].replace(to_replace='No', value=0)
df['RainTomorrow'] = df['RainTomorrow'].replace(to_replace='Yes', value=1)

# Split data into train and test sets
y = df['RainTomorrow']
X = df.drop(['RainTomorrow'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, random_state=1, stratify=y)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=0)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
y_train_res.value_counts().plot(kind="bar", title="Rain Tomorrow Distribution After SMOTE", rot=0)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# SVC Hyperparameter tuning using GridSearchCV
n_cpus = multiprocessing.cpu_count()

parameters = {'C': [0.5, 1, 5, 10]}
svc = SVC(kernel='rbf', random_state=0)

clf = GridSearchCV(svc, parameters, n_jobs=n_cpus)
clf.fit(X_train_scaled, y_train_res)

results_df = pd.DataFrame(clf.cv_results_)
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("name")
print(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])

# Train SVC model
svc = SVC(C=5, kernel='rbf', probability=True, random_state=0)
svc.fit(X_train_scaled, y_train_res)

# Make predictions with SVC
svc_y_pred_prob = svc.predict_proba(X_test_scaled)[:,1]
svc_y_pred = svc.predict(X_test_scaled)

# Evaluate SVC
print('Accuracy score:', accuracy_score(y_test, svc_y_pred))
print('Classifcation report:')
print(classification_report(y_test, svc_y_pred, digits=4))

# Confusion Matrix for SVC
cmat = confusion_matrix(y_test, svc_y_pred)
cmatDisplay = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=svc.classes_)
cmatDisplay.plot()
plt.show()

# ROC Curve for SVC
fpr, tpr, thresholds = roc_curve(y_test, svc_y_pred_prob, pos_label=1)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--')
print('ROC_AUC:', roc_auc_score(y_test, svc_y_pred_prob))

# Random Forest Hyperparameter tuning
parameters = {'criterion': ['gini', 'entropy'], 'n_estimators': [100, 200, 300]}
random_forest = RandomForestClassifier(random_state=0)

clf = GridSearchCV(random_forest, parameters, n_jobs=n_cpus)
clf.fit(X_train_scaled, y_train_res)

results_df = pd.DataFrame(clf.cv_results_)
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("name")
print(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])

# Train Random Forest model
random_forest = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
random_forest.fit(X_train_scaled, y_train_res)

# Make predictions with Random Forest
random_forest_y_pred_prob = random_forest.predict_proba(X_test_scaled)[:,1]
random_forest_y_pred = random_forest.predict(X_test_scaled)

# Evaluate Random Forest
print('Accuracy score:', accuracy_score(y_test, random_forest_y_pred))
print('Classifcation report:')
print(classification_report(y_test, random_forest_y_pred, digits=4))

# Confusion Matrix for Random Forest
cmat = confusion_matrix(y_test, random_forest_y_pred)
cmatDisplay = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=random_forest.classes_)
cmatDisplay.plot()
plt.show()

# ROC Curve for Random Forest
fpr, tpr, thresholds = roc_curve(y_test, random_forest_y_pred_prob, pos_label=1)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--')
print('ROC_AUC:', roc_auc_score(y_test, random_forest_y_pred_prob))

# AdaBoost Hyperparameter tuning
parameters = {'learning_rate': [0.5, 1, 1.5, 2, 5], 'n_estimators': [100, 200, 300]}
adaboost = AdaBoostClassifier(algorithm='SAMME', random_state=0)

clf = GridSearchCV(adaboost, parameters, n_jobs=n_cpus)
clf.fit(X_train_scaled, y_train_res)

results_df = pd.DataFrame(clf.cv_results_)
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("name")
print(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])

# Train AdaBoost model
adaboost = AdaBoostClassifier(n_estimators=300, learning_rate=1, algorithm='SAMME', random_state=0)
adaboost.fit(X_train_scaled, y_train_res)

# Make predictions with AdaBoost
adaboost_y_pred_prob = adaboost.predict_proba(X_test_scaled)[:,1]
adaboost_y_pred = adaboost.predict(X_test_scaled)

# Evaluate AdaBoost
print('Accuracy score:', accuracy_score(y_test, adaboost_y_pred))
print('Classifcation report:')
print(classification_report(y_test, adaboost_y_pred, digits=4))

# Confusion Matrix for AdaBoost
cmat = confusion_matrix(y_test, adaboost_y_pred)
cmatDisplay = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=adaboost.classes_)
cmatDisplay.plot()
plt.show()

# ROC Curve for AdaBoost
fpr, tpr, thresholds = roc_curve(y_test, adaboost_y_pred_prob, pos_label=1)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--')
print('ROC_AUC:', roc_auc_score(y_test, adaboost_y_pred_prob))

# XGBoost Hyperparameter tuning
parameters = {'learning_rate': [0.1, 0.2, 0.3, 0.5], 'n_estimators': [100, 200, 300]}
xgboost = XGBClassifier(random_state=0)

clf = GridSearchCV(xgboost, parameters, n_jobs=n_cpus)
clf.fit(X_train_scaled, y_train_res)

results_df = pd.DataFrame(clf.cv_results_)
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("name")
print(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])

# Train XGBoost model
xgboost = XGBClassifier(n_estimators=300, learning_rate=0.2, random_state=0)
xgboost.fit(X_train_scaled, y_train_res)

# Make predictions with XGBoost
xgboost_y_pred_prob = xgboost.predict_proba(X_test_scaled)[:,1]
xgboost_y_pred = xgboost.predict(X_test_scaled)

# Evaluate XGBoost
print('Accuracy score:', accuracy_score(y_test, xgboost_y_pred))
print('Classifcation report:')
print(classification_report(y_test, xgboost_y_pred, digits=4))

# Confusion Matrix for XGBoost
cmat = confusion_matrix(y_test, xgboost_y_pred)
cmatDisplay = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=xgboost.classes_)
cmatDisplay.plot()
plt.show()

# ROC Curve for XGBoost
fpr, tpr, thresholds = roc_curve(y_test, xgboost_y_pred_prob, pos_label=1)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--')
print('ROC_AUC:', roc_auc_score(y_test, xgboost_y_pred_prob))
