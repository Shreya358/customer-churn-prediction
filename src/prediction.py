"""
prediction.py
-------------
This module takes a NEW customer's details as input and predicts whether
they are likely to churn or not, using the saved best model.

This is the "production" piece of the project: in a real company, this
function would be called every time a customer service rep or a CRM
system wants to check a customer's risk level.

How it works:
1. Load the saved label encoders, scaler, feature column order, and best model.
2. Take the new customer's raw input (as a dictionary).
3. Apply the EXACT same encoding and scaling used during training.
   (This is critical -- if encoding doesn't match training, predictions
   will be wrong or the code will crash.)
4. Return the prediction (Churn / No Churn) and the probability score.
"""

import pandas as pd
import joblib

MODELS_DIR = "models"


def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    encoders = joblib.load(f"{MODELS_DIR}/label_encoders.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
    return model, encoders, scaler, feature_columns


def predict_churn(customer: dict, model=None, encoders=None, scaler=None, feature_columns=None):
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer : dict
        Raw customer details, e.g.
        {
            "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes",
            "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 75.5,
            "TotalCharges": 380.0
        }

    Returns
    -------
    dict with keys: prediction ("Churn"/"No Churn"), probability (float 0-1)
    """
    if model is None:
        model, encoders, scaler, feature_columns = load_artifacts()

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Build a single-row DataFrame in the exact column order used at training time
    row = {}
    for col in feature_columns:
        if col in numeric_cols:
            row[col] = customer[col]
        else:
            # Apply the SAME label encoder fitted during training
            le = encoders[col]
            row[col] = le.transform([customer[col]])[0]

    X_new = pd.DataFrame([row], columns=feature_columns)

    # Scale numeric columns using the SAME scaler fitted during training
    X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])

    proba = model.predict_proba(X_new)[0][1]   # probability of class "1" = Churn
    prediction = "Churn" if proba >= 0.5 else "No Churn"

    return {
        "prediction": prediction,
        "probability": round(float(proba), 4)
    }


if __name__ == "__main__":
    # Example test customer: new, high-paying, month-to-month -> expect HIGH churn risk
    sample_high_risk = {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "No",
        "Dependents": "No", "tenure": 1, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0,
        "TotalCharges": 95.0
    }

    # Example test customer: long-term, low-paying, 2-year contract -> expect LOW churn risk
    sample_low_risk = {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes",
        "Dependents": "Yes", "tenure": 60, "PhoneService": "Yes",
        "MultipleLines": "Yes", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0
    }

    model, encoders, scaler, feature_columns = load_artifacts()

    result1 = predict_churn(sample_high_risk, model, encoders, scaler, feature_columns)
    print("High-risk sample customer ->", result1)

    result2 = predict_churn(sample_low_risk, model, encoders, scaler, feature_columns)
    print("Low-risk sample customer  ->", result2)
