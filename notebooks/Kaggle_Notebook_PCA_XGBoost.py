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
from sklearn.decomposition import PCA
import multiprocessing
import joblib

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

# Apply PCA for dimensionality reduction (90% variance retention)
pca = PCA(n_components=0.9)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Save the PCA-reduced dataset
X_train_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
X_test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])

# Concatenate the reduced datasets with the target variable
train_data_pca = pd.concat([X_train_pca_df, y_train_res.reset_index(drop=True)], axis=1)
test_data_pca = pd.concat([X_test_pca_df, y_test.reset_index(drop=True)], axis=1)

# Save the datasets as CSV files
train_data_pca.to_csv("weatherAUS_pca_train.csv", index=False)
test_data_pca.to_csv("weatherAUS_pca_test.csv", index=False)

# XGBoost Hyperparameter tuning
parameters = {'learning_rate': [0.1, 0.2, 0.3, 0.5], 'n_estimators': [100, 200, 300]}
xgboost = XGBClassifier(random_state=0)

# Use multiprocessing to speed up GridSearchCV
n_cpus = multiprocessing.cpu_count()

clf = GridSearchCV(xgboost, parameters, n_jobs=n_cpus)
clf.fit(X_train_pca, y_train_res)

results_df = pd.DataFrame(clf.cv_results_)
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("name")
print(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])

# Train XGBoost model
xgboost = XGBClassifier(n_estimators=300, learning_rate=0.2, random_state=0)
xgboost.fit(X_train_pca, y_train_res)

# Make predictions with XGBoost
xgboost_y_pred_prob = xgboost.predict_proba(X_test_pca)[:,1]
xgboost_y_pred = xgboost.predict(X_test_pca)

# Evaluate XGBoost
print('Accuracy score:', accuracy_score(y_test, xgboost_y_pred))
print('Classification report:')
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

# Save the trained XGBoost model
joblib.dump(xgboost, "xgboost_model.pkl")

print("Model saved successfully!")
