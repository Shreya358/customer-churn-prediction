"""
run_pipeline.py
-----------------
Master script that runs the ENTIRE machine learning pipeline in the
correct order, end to end. Useful for:
- Quickly re-running everything after a fresh git clone
- Demonstrating the full pipeline live in an interview

Run with:  python3 src/run_pipeline.py

NOTE: This does NOT include the MySQL load step, since that requires a
running MySQL server. Run that separately:
    1. Start MySQL:        service mysql start   (Linux)  /  net start mysql (Windows, as admin)
    2. mysql -u root < sql/schema.sql
    3. python3 src/load_to_mysql.py
"""

import subprocess
import sys

STEPS = [
    ("Data Cleaning", "src/preprocessing.py"),
    ("Exploratory Data Analysis", "src/eda.py"),
    ("Feature Engineering", "src/feature_engineering.py"),
    ("Model Training & Comparison", "src/model_training.py"),
    ("Feature Importance", "src/feature_importance.py"),
    ("Prediction Module Test", "src/prediction.py"),
]


def run_step(step_name: str, script_path: str):
    print("\n" + "=" * 60)
    print(f"STEP: {step_name}")
    print("=" * 60)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"\n FAILED at step: {step_name}")
        sys.exit(1)


if __name__ == "__main__":
    print("Starting Customer Churn Prediction Pipeline...\n")

    for step_name, script_path in STEPS:
        run_step(step_name, script_path)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("""
Next steps:
  1. Launch the dashboard:   streamlit run src/streamlit_app.py
  2. Load data into MySQL:  python3 src/load_to_mysql.py  (after starting MySQL)
  3. Run SQL queries:       mysql -u root customer_churn_db < sql/business_queries.sql
""")
