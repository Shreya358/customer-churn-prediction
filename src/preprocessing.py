"""
preprocessing.py
----------------
This script handles DATA CLEANING for the Customer Churn project.

What it does (in plain English, for interview explanation):
1. Loads the raw IBM Telco Customer Churn CSV.
2. Fixes the 'TotalCharges' column, which is stored as TEXT instead of a
   number, and contains 11 blank values (customers with 0 months tenure).
3. Removes duplicate customer records (if any).
4. Converts 'Yes'/'No' style text columns into a clean, consistent format.
5. Saves a clean version of the dataset to data/processed/.

Why this step matters (business reasoning):
Machine learning models cannot work with messy data. If 'TotalCharges' stays
as text, Python will treat it like a string and we can't do any math on it
(like averages or correlations). Cleaning data is usually 60-70% of a real
Data Analyst/Data Scientist's job, so interviewers care a LOT about this step.
"""

import pandas as pd
import os

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_churn_data.csv"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file into a pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco churn dataset.
    Returns a cleaned DataFrame ready for EDA and modeling.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Fix TotalCharges column
    # ---------------------------------------------------------
    # This column should be numeric, but it is stored as text and has
    # 11 rows where the value is just a blank space (" ").
    # These blank rows all belong to customers with tenure = 0,
    # meaning they are brand-new customers who haven't been billed yet.
    # We convert it to numeric and fill those blanks with 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_total_charges = df["TotalCharges"].isnull().sum()
    print(f"Found {missing_total_charges} blank TotalCharges values -> filling with 0")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # ---------------------------------------------------------
    # 2. Remove duplicate rows (based on customerID)
    # ---------------------------------------------------------
    before = df.shape[0]
    df = df.drop_duplicates(subset="customerID")
    after = df.shape[0]
    print(f"Removed {before - after} duplicate customer rows")

    # ---------------------------------------------------------
    # 3. Standardize column names (lowercase, no spaces)
    # ---------------------------------------------------------
    df.columns = [c.strip() for c in df.columns]

    # ---------------------------------------------------------
    # 4. Clean SeniorCitizen column
    # ---------------------------------------------------------
    # SeniorCitizen is stored as 0/1, but every other Yes/No column
    # is stored as text. We convert it to "Yes"/"No" for consistency,
    # which makes EDA charts and grouping easier to read.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # ---------------------------------------------------------
    # 5. Strip whitespace from all text/object columns
    # ---------------------------------------------------------
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()

    # ---------------------------------------------------------
    # 6. Final missing value check
    # ---------------------------------------------------------
    null_counts = df.isnull().sum().sum()
    print(f"Remaining missing values after cleaning: {null_counts}")

    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save cleaned data to the processed folder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to: {path}")


if __name__ == "__main__":
    raw_df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_data(raw_df)
    save_data(cleaned_df, PROCESSED_DATA_PATH)
    print("\nPreprocessing complete.")
    print(cleaned_df.head())
