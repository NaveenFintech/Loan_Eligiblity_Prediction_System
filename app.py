from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# -----------------------------
# Load Models Safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")

logistic = joblib.load(os.path.join(MODEL_PATH, "logistic.pkl"))
rf = joblib.load(os.path.join(MODEL_PATH, "rf.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        input_values = [
            float(form_data.get("Gender", 0)),
            float(form_data.get("Married", 0)),
            float(form_data.get("Dependents", 0)),
            float(form_data.get("Education", 0)),
            float(form_data.get("Self_Employed", 0)),
            float(form_data.get("ApplicantIncome", 0)),
            float(form_data.get("CoapplicantIncome", 0)),
            float(form_data.get("LoanAmount", 0)),
            float(form_data.get("Loan_Amount_Term", 0)),
            float(form_data.get("Credit_History", 0)),
            float(form_data.get("Property_Area", 0))
        ]

        features = np.array([input_values])
        features = scaler.transform(features)

        # Use Random Forest for prediction
        prediction = rf.predict(features)[0]
        probability = rf.predict_proba(features)[0][1] * 100

        # Risk classification
        if probability > 75:
            risk = "Low Risk 🟢"
        elif probability > 50:
            risk = "Medium Risk 🟡"
        else:
            risk = "High Risk 🔴"

        result = "Loan Approved ✅" if prediction == 1 else "Loan Not Approved ❌"

        return render_template(
            "result.html",
            result=result,
            probability=round(probability, 2),
            risk=risk
        )

    except Exception as e:
        return render_template(
            "result.html",
            result="Error occurred",
            probability=0,
            risk=str(e)
        )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json.get("features")
        features = scaler.transform([data])

        prediction = rf.predict(features)[0]
        probability = rf.predict_proba(features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------
# Production Run (Render Ready)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)