"""
model_training.py
------------------
Trains and compares 3 Machine Learning models for churn prediction:
1. Logistic Regression  - simple, fast, highly interpretable baseline
2. Decision Tree         - easy to visualize and explain decision rules
3. Random Forest         - combines many decision trees, usually best accuracy

Why these 3 models (interview explanation):
- Logistic Regression is the natural baseline for any binary classification
  problem ("will churn" vs "won't churn"). If a complex model can't beat
  this, the complexity isn't worth it.
- Decision Tree shows clear IF-THEN business rules, e.g.
  "IF Contract = Month-to-month AND tenure < 6 THEN High Risk".
- Random Forest is an ensemble of many Decision Trees. It usually performs
  best because it reduces overfitting by averaging many trees together.

Evaluation Metrics explained:
- Accuracy: % of total predictions that were correct. Can be misleading
  here because only ~26.5% of customers churn (class imbalance).
- Precision: Of all customers we PREDICTED would churn, how many actually did?
  High precision = fewer wasted retention offers on customers who would've stayed anyway.
- Recall: Of all customers who ACTUALLY churned, how many did we catch?
  High recall = we don't miss customers who are about to leave.
- F1 Score: harmonic mean of Precision and Recall -- a balanced single score.
- ROC-AUC: measures how well the model separates churners from non-churners
  across all possible thresholds, regardless of class imbalance.

For a churn use case, RECALL is usually prioritized: it's more costly to
miss a customer who churns than to send a retention offer to one who
would have stayed anyway.
"""

import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
IMAGES_DIR = "images"


def load_split_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    """Calculate and print all key metrics for one model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k:<10}: {v:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return metrics, cm, y_proba


def plot_confusion_matrix(cm, name, filename):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}", fontweight="bold")
    plt.colorbar()
    classes = ["No Churn", "Churn"]
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      fontsize=14, fontweight="bold",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/{filename}", dpi=150)
    plt.close()


def plot_model_comparison(results_df):
    """Bar chart comparing all models across all metrics."""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        plt.bar(x + i * width, values, width, label=row["Model"])

    plt.xticks(x + width, metrics_to_plot)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/09_model_comparison.png", dpi=150)
    plt.close()
    print("\nSaved: 09_model_comparison.png")


if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    }

    all_results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics, cm, y_proba = evaluate_model(name, model, X_test, y_test)
        all_results.append(metrics)

        filename = f"cm_{name.lower().replace(' ', '_')}.png"
        plot_confusion_matrix(cm, name, filename)

    results_df = pd.DataFrame(all_results)
    print("\n=== Final Model Comparison Table ===")
    print(results_df.to_string(index=False))

    plot_model_comparison(results_df)

    # ---------------------------------------------------------
    # Select best model based on F1 Score (balances precision & recall,
    # which matters most for an imbalanced churn dataset).
    # ---------------------------------------------------------
    best_model_name = results_df.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest model selected: {best_model_name}")

    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(best_model_name, f"{MODELS_DIR}/best_model_name.pkl")
    results_df.to_csv(f"{PROCESSED_DIR}/model_comparison_results.csv", index=False)

    print(f"Saved best model to {MODELS_DIR}/best_model.pkl")
