from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load models
logistic = joblib.load("models/logistic.pkl")
rf = joblib.load("models/rf.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    input_values = [
        float(form_data["Gender"]),
        float(form_data["Married"]),
        float(form_data["Dependents"]),
        float(form_data["Education"]),
        float(form_data["Self_Employed"]),
        float(form_data["ApplicantIncome"]),
        float(form_data["CoapplicantIncome"]),
        float(form_data["LoanAmount"]),
        float(form_data["Loan_Amount_Term"]),
        float(form_data["Credit_History"]),
        float(form_data["Property_Area"])
    ]

    features = np.array([input_values])
    features = scaler.transform(features)

    # Use Random Forest for prediction
    prediction = rf.predict(features)[0]
    probability = rf.predict_proba(features)[0][1] * 100

    if probability > 75:
        risk = "Low Risk"
    elif probability > 50:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    result = "Loan Approved" if prediction == 1 else "Loan Not Approved"

    return render_template(
        "result.html",
        result=result,
        probability=round(probability, 2),
        risk=risk
    )

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json["features"]
    features = scaler.transform([data])
    prediction = rf.predict(features)[0]
    probability = rf.predict_proba(features)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)