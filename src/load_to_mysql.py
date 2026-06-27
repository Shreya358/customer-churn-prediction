"""
load_to_mysql.py
------------------
Loads the cleaned churn dataset from CSV into the MySQL database.

Run this AFTER:
1. mysql -u root < sql/schema.sql      (creates the database + table)
2. python3 src/preprocessing.py        (creates the cleaned CSV)

This script uses SQLAlchemy + mysql-connector-python to connect to
MySQL and bulk-insert the cleaned data with pandas' to_sql() function.

Update DB_USER / DB_PASSWORD / DB_HOST below to match your own MySQL setup.
"""

import os
import pandas as pd
import mysql.connector

# ----------------------------------------------------------------------
# Database connection settings — EDIT THESE for your own MySQL setup
# or configure the values via environment variables before running.
# Example (PowerShell):
#   $env:DB_USER = 'your_user'
#   $env:DB_PASSWORD = 'your_password'
#   $env:DB_HOST = 'localhost'
#   $env:DB_PORT = '3306'
#   $env:DB_NAME = 'customer_churn_db'
# ----------------------------------------------------------------------
DB_USER = os.getenv("DB_USER", "churn_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "customer_churn_db")

CLEANED_DATA_PATH = "data/processed/cleaned_churn_data.csv"

# Map pandas column names -> MySQL column names (snake_case)
COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}


def get_connection():
    return mysql.connector.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )


def load_to_mysql():
    df = pd.read_csv(CLEANED_DATA_PATH)
    df = df.rename(columns=COLUMN_MAP)

    insert_columns = ", ".join(df.columns)
    placeholders = ", ".join(["%s"] * len(df.columns))
    insert_sql = f"INSERT INTO customers ({insert_columns}) VALUES ({placeholders})"

    rows = [tuple(row) for row in df.to_numpy()]

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM customers")
        cursor.executemany(insert_sql, rows)
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    print(f"Loaded {df.shape[0]} rows into {DB_NAME}.customers")


if __name__ == "__main__":
    load_to_mysql()
