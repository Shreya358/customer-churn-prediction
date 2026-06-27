# 📊 Customer Churn Prediction & Retention Analytics

A complete, end-to-end Data Science project that predicts which telecom
customers are likely to **churn** (cancel their subscription), and turns
that prediction into actionable business insights through SQL analysis,
an interactive Streamlit dashboard, and a Power BI report.

Built using the real **IBM Telco Customer Churn dataset** (7,043 customers).

---

## 📌 Project Overview

Customer churn — when a paying customer stops using a company's service —
is one of the most expensive problems in subscription-based businesses
(telecom, SaaS, banking, streaming, etc.). It is far cheaper to **retain**
an existing customer than to acquire a new one.

This project answers three business questions:
1. **Who** is likely to churn next?
2. **Why** are they likely to churn (which factors drive it)?
3. **What** should the business do about it?

---

## 🎯 Objectives

- Clean and preprocess real-world, messy customer data
- Perform exploratory data analysis (EDA) to uncover churn patterns
- Engineer features suitable for machine learning
- Train and compare 3 ML models (Logistic Regression, Decision Tree, Random Forest)
- Select the best model using proper evaluation metrics (not just accuracy)
- Build a live prediction tool for new customers
- Analyze the data using 18 business-focused SQL queries (MySQL)
- Build an interactive Streamlit dashboard for technical demo
- Build a Power BI dashboard for business stakeholders

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Database | MySQL 8.0 |
| Web Dashboard | Streamlit |
| BI Dashboard | Power BI |
| Version Control | Git & GitHub |

---

## 📂 Dataset

**Source:** IBM Telco Customer Churn dataset (industry-standard, widely
used in churn-prediction case studies)

- **Rows:** 7,043 customers
- **Columns:** 21 (20 features + target)
- **Target variable:** `Churn` (Yes / No)

| Column | Description |
|---|---|
| customerID | Unique customer identifier |
| gender | Male / Female |
| SeniorCitizen | Whether the customer is a senior citizen |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Number of months the customer has stayed |
| PhoneService | Whether the customer has phone service |
| MultipleLines | Whether the customer has multiple phone lines |
| InternetService | DSL / Fiber optic / No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport | Add-on services |
| StreamingTV, StreamingMovies | Streaming add-ons |
| Contract | Month-to-month / One year / Two year |
| PaperlessBilling | Yes / No |
| PaymentMethod | Electronic check / Mailed check / Bank transfer / Credit card |
| MonthlyCharges | Current monthly bill amount |
| TotalCharges | Total amount billed to the customer so far |
| Churn | **Target** — did the customer leave? |

---

## 📁 Folder Structure

```
customer_churn_prediction/
│
├── data/
│   ├── raw/                         # Original IBM Telco CSV
│   └── processed/                   # Cleaned data, train/test splits, results
│
├── src/
│   ├── preprocessing.py             # Data cleaning
│   ├── eda.py                       # Exploratory Data Analysis + charts
│   ├── feature_engineering.py       # Encoding, scaling, train/test split
│   ├── model_training.py            # Trains & compares 3 ML models
│   ├── feature_importance.py        # Explains which features matter most
│   ├── prediction.py                # Predicts churn for a new customer
│   ├── load_to_mysql.py             # Loads cleaned data into MySQL
│   ├── run_pipeline.py              # Runs the full ML pipeline in one command
│   └── streamlit_app.py             # Interactive web dashboard
│
├── sql/
│   ├── schema.sql                   # MySQL table schema
│   └── business_queries.sql         # 18 business SQL queries
│
├── powerbi/
│   ├── churn_data_for_powerbi.csv   # Data ready for Power BI import
│   └── POWERBI_SETUP_GUIDE.md       # Step-by-step dashboard build guide
│
├── models/                          # Saved trained model + encoders/scaler
├── images/                          # All generated charts (EDA, model comparison, etc.)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full ML pipeline (one command)
```bash
python src/run_pipeline.py
```
This runs data cleaning → EDA → feature engineering → model training →
feature importance → a prediction test, in order, and saves all outputs
(charts to `images/`, trained model to `models/`).

### 5. Launch the Streamlit dashboard
```bash
streamlit run src/streamlit_app.py
```
Opens at `http://localhost:8501` in your browser.

### 6. Set up MySQL (for the SQL analysis section)
```bash
# Start MySQL first (varies by OS):
#   Linux:   sudo service mysql start
#   macOS:   brew services start mysql
#   Windows: net start mysql

mysql -u root < sql/schema.sql
python src/load_to_mysql.py
mysql -u root customer_churn_db < sql/business_queries.sql
```
> ⚠️ Edit the `DB_USER` / `DB_PASSWORD` values in `src/load_to_mysql.py`
> to match your own MySQL credentials before running.

### 7. Build the Power BI dashboard
See `powerbi/POWERBI_SETUP_GUIDE.md` for full step-by-step instructions
using `powerbi/churn_data_for_powerbi.csv`.

---

## 🔄 Project Workflow

```
Raw CSV Data
     │
     ▼
Data Cleaning (preprocessing.py)
     │
     ▼
Exploratory Data Analysis (eda.py)
     │
     ▼
Feature Engineering (feature_engineering.py)
     │
     ▼
Model Training & Comparison (model_training.py)
     │
     ▼
Best Model Selected → saved as models/best_model.pkl
     │
     ├──► Feature Importance Analysis
     ├──► Prediction Module (for new customers)
     ├──► Streamlit Dashboard (interactive demo)
     ├──► MySQL Database + 18 Business SQL Queries
     └──► Power BI Dashboard (business stakeholder view)
```

---

## 📈 Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 80.0% | 64.5% | 54.8% | **59.2%** | 84.0% |
| Decision Tree | 78.4% | 60.4% | 54.5% | 57.3% | 82.5% |
| Random Forest | 79.6% | 64.7% | 50.5% | 56.8% | 83.6% |

**Best model: Logistic Regression** — selected based on F1 Score, since
the dataset is imbalanced (~26.5% churners) and F1 balances precision
(avoiding wasted retention offers on customers who'd stay anyway) against
recall (not missing customers who are about to leave).

> **Talking point for interviews:** Logistic Regression beating the more
> "complex" models is a real and common result — it shows the relationship
> between churn and these features is mostly linear/additive, and a simpler,
> faster, more interpretable model wins here. This is a *good* finding, not
> a failure — knowing when NOT to reach for a complex model is a sign of
> good judgment, not a limitation.

### Key Business Insights
- Overall churn rate: **26.5%**
- Month-to-month customers churn at **42.7%** vs **2.8%** for two-year contracts
- Customers with **tenure ≤ 12 months churn at 47.4%** vs **9.5%** for 4+ year customers
- **Electronic check** payers churn at **45.3%** — the highest of any payment method
- Customers without **Tech Support** or **Online Security** churn roughly **3x more**
- Estimated **30.5%** of monthly recurring revenue is currently at risk from churned customers

### Top 5 Churn Drivers (Feature Importance)
1. Tenure
2. Phone Service
3. Monthly Charges
4. Contract Type
5. Total Charges

---

## 🖼️ Screenshots

> Charts are auto-generated in the `images/` folder when you run the pipeline.

- `images/01_churn_distribution.png` — Overall churn split
- `images/03_contract_vs_churn.png` — Churn rate by contract type
- `images/09_model_comparison.png` — Model performance comparison
- `images/10_feature_importance.png` — Feature importance ranking
- *(Streamlit dashboard screenshot — add your own after running locally)*
- *(Power BI dashboard screenshot — add your own after building it)*

---

## 🚀 Future Improvements

- Add SMOTE or class-weighting to better handle class imbalance
- Try hyperparameter tuning (GridSearchCV) for further performance gains
- Add SHAP values for more detailed, per-customer explainability
- Deploy the Streamlit app publicly (Streamlit Community Cloud)
- Add a model monitoring/retraining pipeline for production use
- A/B test actual retention campaigns based on model predictions

---

## 💼 Resume Highlights

> **Customer Churn Prediction & Retention Analytics** | Python, Scikit-learn, MySQL, Streamlit, Power BI
> - Built an end-to-end churn prediction system on 7,000+ telecom customer
>   records, cleaning real-world messy data and engineering features for ML.
> - Trained and compared 3 classification models, selecting the best
>   performer using F1 Score and ROC-AUC to handle class imbalance.
> - Wrote 18 business-focused SQL queries in MySQL to surface churn drivers
>   across contract type, payment method, and customer tenure segments.
> - Built an interactive Streamlit dashboard enabling live churn-risk
>   prediction for new customers with a 19-field input form.
> - Designed a Power BI dashboard with KPI cards, slicers, and segmentation
>   views for non-technical business stakeholders.

---

## 🎤 Interview Questions & Answers

See **Interview Preparation** section below — 50 questions across Python,
Pandas, SQL, Machine Learning, Business Analytics, Power BI, Streamlit,
and project architecture, each with a concise model answer.

---

## 📜 License

This project uses the publicly available IBM Telco Customer Churn dataset
for educational and portfolio purposes.
