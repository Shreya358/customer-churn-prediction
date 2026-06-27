"""
feature_engineering.py
-----------------------
Prepares the cleaned data for Machine Learning.

What this script does:
1. Drops columns that don't help prediction (customerID is just an ID).
2. Converts the target column 'Churn' into 0/1 (machines need numbers).
3. Encodes all categorical (text) columns into numbers using Label Encoding,
   because ML models like Logistic Regression and Decision Trees cannot
   read text directly.
4. Scales numeric columns (tenure, MonthlyCharges, TotalCharges) so that
   no single feature dominates just because its numbers are bigger.
5. Splits the data into Training (80%) and Testing (20%) sets.

Why this matters (interview explanation):
- "Garbage in, garbage out" -- a model is only as good as the features
  you feed it.
- Label Encoding: turns categories like 'DSL', 'Fiber optic', 'No' into
  0, 1, 2 so the model can do math on them.
- Scaling: Logistic Regression is sensitive to feature scale (a column
  with values 0-9000 like TotalCharges would dominate a column with
  values 0-1 like SeniorCitizen). Tree-based models don't strictly need
  scaling, but we scale anyway to keep one consistent pipeline for all
  models, which is a common real-world practice.
- Train/Test split: we train the model on 80% of customers and test it on
  the remaining 20% it has NEVER seen, to fairly check if it generalizes
  to new customers instead of just memorizing the training data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

CLEANED_DATA_PATH = "data/processed/cleaned_churn_data.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEANED_DATA_PATH)


def engineer_features(df: pd.DataFrame):
    """
    Transforms the cleaned dataframe into model-ready X (features) and y (target).
    Returns: X (DataFrame), y (Series), encoders (dict), scaler (StandardScaler)
    """
    df = df.copy()

    # ---------------------------------------------------------
    # 1. Drop customerID - it's just a unique identifier with
    #    zero predictive value, like a row number.
    # ---------------------------------------------------------
    df = df.drop(columns=["customerID"])

    # ---------------------------------------------------------
    # 2. Encode target variable: Churn -> 0/1
    # ---------------------------------------------------------
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # ---------------------------------------------------------
    # 3. Identify categorical columns (everything except the
    #    numeric ones and the target)
    # ---------------------------------------------------------
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ["Churn"]]

    # ---------------------------------------------------------
    # 4. Label Encode categorical columns
    #    Example: Contract -> Month-to-month=0, One year=1, Two year=2
    # ---------------------------------------------------------
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # ---------------------------------------------------------
    # 5. Separate features (X) and target (y)
    # ---------------------------------------------------------
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # ---------------------------------------------------------
    # 6. Scale numeric columns
    # ---------------------------------------------------------
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, encoders, scaler


def split_data(X: pd.DataFrame, y: pd.Series):
    """Split into 80% train / 20% test, keeping the same churn ratio in both (stratify)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} customers")
    print(f"Testing set:  {X_test.shape[0]} customers")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    data = load_clean_data()
    X, y, encoders, scaler = engineer_features(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save everything needed downstream (model training + prediction module)
    joblib.dump(encoders, f"{MODELS_DIR}/label_encoders.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(list(X.columns), f"{MODELS_DIR}/feature_columns.pkl")

    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)

    print("\nFeature engineering complete.")
    print("Final feature columns:", list(X.columns))
