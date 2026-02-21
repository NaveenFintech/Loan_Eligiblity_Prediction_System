import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# Load dataset
data = pd.read_csv("dataset/loan-train.csv")

# Drop Loan_ID
data.drop("Loan_ID", axis=1, inplace=True)

# Fix Dependents column
data["Dependents"] = data["Dependents"].replace("3+", 3)
data["Dependents"] = pd.to_numeric(data["Dependents"])

# Fill missing values
for col in data.columns:
    if data[col].dtype == "object":
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Split
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
logistic = LogisticRegression(max_iter=5000)
logistic.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Accuracy
log_acc = accuracy_score(y_test, logistic.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print("Logistic Accuracy:", log_acc)
print("Random Forest Accuracy:", rf_acc)

# Save models
joblib.dump(logistic, "models/logistic.pkl")
joblib.dump(rf, "models/rf.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Models saved successfully.")