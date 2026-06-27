"""
eda.py
------
Exploratory Data Analysis (EDA) for the Customer Churn project.

What this script does:
Generates a set of Matplotlib charts that help us UNDERSTAND the data
before we build any model. Each chart answers a specific business question.

Why EDA matters (for interview explanation):
EDA helps us find patterns, spot problems, and form a hypothesis about
WHY customers churn, before we touch any machine learning model. A good
Data Analyst always looks at the data first.

All charts are saved as PNG files inside the images/ folder, so they can be
reused in the README, the Streamlit dashboard, and your PPT/Notion docs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

CLEANED_DATA_PATH = "data/processed/cleaned_churn_data.csv"
IMAGES_DIR = "images"

# Use a clean, professional chart style
plt.style.use("seaborn-v0_8-whitegrid")
COLOR_NO_CHURN = "#2E86AB"
COLOR_CHURN = "#E63946"


def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEANED_DATA_PATH)


def churn_distribution(df: pd.DataFrame):
    """
    Chart 1: Overall churn distribution (class balance).
    Business question: Out of all customers, how many actually leave?
    """
    counts = df["Churn"].value_counts()
    plt.figure(figsize=(6, 5))
    bars = plt.bar(counts.index, counts.values, color=[COLOR_NO_CHURN, COLOR_CHURN])
    plt.title("Customer Churn Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    for bar in bars:
        height = bar.get_height()
        pct = height / counts.sum() * 100
        plt.text(bar.get_x() + bar.get_width() / 2, height + 30,
                  f"{height}\n({pct:.1f}%)", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/01_churn_distribution.png", dpi=150)
    plt.close()
    print("Saved: 01_churn_distribution.png  -> About 26.5% of customers churn (class imbalance)")


def gender_vs_churn(df: pd.DataFrame):
    """
    Chart 2: Gender vs Churn.
    Business question: Does gender affect whether a customer leaves?
    """
    ct = pd.crosstab(df["gender"], df["Churn"])
    ct.plot(kind="bar", color=[COLOR_NO_CHURN, COLOR_CHURN], figsize=(6, 5))
    plt.title("Gender vs Churn", fontsize=14, fontweight="bold")
    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/02_gender_vs_churn.png", dpi=150)
    plt.close()
    print("Saved: 02_gender_vs_churn.png  -> Gender shows almost no impact on churn")


def contract_vs_churn(df: pd.DataFrame):
    """
    Chart 3: Contract Type vs Churn.
    Business question: Are month-to-month customers more likely to leave?
    """
    ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", color=[COLOR_NO_CHURN, COLOR_CHURN], figsize=(7, 5))
    plt.title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage of Customers (%)")
    plt.xticks(rotation=0)
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/03_contract_vs_churn.png", dpi=150)
    plt.close()
    print("Saved: 03_contract_vs_churn.png  -> Month-to-month customers churn far more than yearly contracts")


def internet_service_vs_churn(df: pd.DataFrame):
    """
    Chart 4: Internet Service Type vs Churn.
    Business question: Does the type of internet service affect churn?
    """
    ct = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", color=[COLOR_NO_CHURN, COLOR_CHURN], figsize=(7, 5))
    plt.title("Churn Rate by Internet Service Type", fontsize=14, fontweight="bold")
    plt.xlabel("Internet Service")
    plt.ylabel("Percentage of Customers (%)")
    plt.xticks(rotation=0)
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/04_internet_service_vs_churn.png", dpi=150)
    plt.close()
    print("Saved: 04_internet_service_vs_churn.png  -> Fiber optic customers churn the most")


def monthly_charges_vs_churn(df: pd.DataFrame):
    """
    Chart 5: Monthly Charges distribution by Churn.
    Business question: Do customers who pay more tend to churn more?
    """
    plt.figure(figsize=(7, 5))
    plt.hist(df[df["Churn"] == "No"]["MonthlyCharges"], bins=30, alpha=0.6,
              label="No Churn", color=COLOR_NO_CHURN)
    plt.hist(df[df["Churn"] == "Yes"]["MonthlyCharges"], bins=30, alpha=0.6,
              label="Churn", color=COLOR_CHURN)
    plt.title("Monthly Charges Distribution by Churn", fontsize=14, fontweight="bold")
    plt.xlabel("Monthly Charges ($)")
    plt.ylabel("Number of Customers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/05_monthly_charges_vs_churn.png", dpi=150)
    plt.close()
    print("Saved: 05_monthly_charges_vs_churn.png  -> Customers with higher monthly charges churn more")


def tenure_distribution(df: pd.DataFrame):
    """
    Chart 6: Tenure distribution by Churn.
    Business question: Do newer customers churn more than long-term ones?
    """
    plt.figure(figsize=(7, 5))
    plt.hist(df[df["Churn"] == "No"]["tenure"], bins=30, alpha=0.6,
              label="No Churn", color=COLOR_NO_CHURN)
    plt.hist(df[df["Churn"] == "Yes"]["tenure"], bins=30, alpha=0.6,
              label="Churn", color=COLOR_CHURN)
    plt.title("Tenure Distribution by Churn", fontsize=14, fontweight="bold")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Number of Customers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/06_tenure_distribution.png", dpi=150)
    plt.close()
    print("Saved: 06_tenure_distribution.png  -> New customers (low tenure) churn the most")


def correlation_heatmap(df: pd.DataFrame):
    """
    Chart 7: Correlation heatmap of numeric features.
    Business question: Which numeric features move together?
    """
    numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
    numeric_df["Churn_numeric"] = (df["Churn"] == "Yes").astype(int)
    corr = numeric_df.corr()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                      color="black", fontsize=9)
    plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/07_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: 07_correlation_heatmap.png  -> Tenure is negatively correlated with churn")


def payment_method_vs_churn(df: pd.DataFrame):
    """
    Chart 8: Payment Method vs Churn.
    Business question: Does HOW a customer pays affect their churn risk?
    """
    ct = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100
    ct.plot(kind="barh", color=[COLOR_NO_CHURN, COLOR_CHURN], figsize=(8, 5))
    plt.title("Churn Rate by Payment Method", fontsize=14, fontweight="bold")
    plt.xlabel("Percentage of Customers (%)")
    plt.ylabel("Payment Method")
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/08_payment_method_vs_churn.png", dpi=150)
    plt.close()
    print("Saved: 08_payment_method_vs_churn.png  -> Electronic check users churn the most")


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    data = load_clean_data()

    print(f"\nRunning EDA on {data.shape[0]} customers...\n")
    churn_distribution(data)
    gender_vs_churn(data)
    contract_vs_churn(data)
    internet_service_vs_churn(data)
    monthly_charges_vs_churn(data)
    tenure_distribution(data)
    correlation_heatmap(data)
    payment_method_vs_churn(data)

    print("\nAll EDA charts saved successfully in the images/ folder.")
