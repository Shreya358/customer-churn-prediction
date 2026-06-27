"""
feature_importance.py
-----------------------
Shows WHICH features matter most in predicting churn, and explains
what that means for the business.

How importance is calculated per model (interview explanation):
- Logistic Regression: uses the absolute value of each feature's
  coefficient. A bigger coefficient (positive or negative) means that
  feature pushes the prediction more strongly toward churn or non-churn.
- Decision Tree / Random Forest: use 'feature_importances_', which
  measures how much each feature reduces impurity (i.e. how useful it
  was for splitting customers into churn vs non-churn groups) across
  the tree(s).

This script automatically detects which model is the "best_model" and
picks the right method.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

MODELS_DIR = "models"
IMAGES_DIR = "images"


def get_feature_importance(model, feature_names):
    """Return a DataFrame of feature -> importance score, sorted descending."""
    if hasattr(model, "feature_importances_"):
        # Decision Tree / Random Forest
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic Regression - use absolute coefficient value
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model type not supported for feature importance.")

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    return df


def plot_feature_importance(df, model_name):
    plt.figure(figsize=(8, 7))
    plt.barh(df["Feature"][::-1], df["Importance"][::-1], color="#2E86AB")
    plt.title(f"Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/10_feature_importance.png", dpi=150)
    plt.close()
    print("Saved: 10_feature_importance.png")


if __name__ == "__main__":
    best_model = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    best_model_name = joblib.load(f"{MODELS_DIR}/best_model_name.pkl")
    feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")

    importance_df = get_feature_importance(best_model, feature_columns)

    print(f"\nFeature Importance ({best_model_name}):")
    print(importance_df.to_string(index=False))

    plot_feature_importance(importance_df, best_model_name)

    importance_df.to_csv("data/processed/feature_importance.csv", index=False)

    print("\nTop 5 business drivers of churn:")
    for i, row in importance_df.head(5).iterrows():
        print(f"  - {row['Feature']}")
