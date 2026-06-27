"""
streamlit_app.py
-----------------
Interactive web dashboard for the Customer Churn Prediction project.

Run with:  streamlit run src/streamlit_app.py

Sections in this dashboard:
1. Project overview / title
2. Dataset overview (basic stats)
3. EDA visualizations (the charts we generated earlier)
4. Model performance comparison
5. Live prediction tool (enter a customer's details, get a prediction)
6. Feature importance
7. Business summary / insights

Why Streamlit (interview explanation):
Streamlit turns a plain Python script into an interactive web app with
almost no extra code -- no HTML/CSS/JS needed. It's the fastest way for a
Data Scientist to demo a model to non-technical stakeholders.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prediction import predict_churn, load_artifacts

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_churn_data.csv"))


@st.cache_resource
def load_model_artifacts():
    return load_artifacts()


@st.cache_data
def load_results():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "model_comparison_results.csv"))


df = load_data()
model, encoders, scaler, feature_columns = load_model_artifacts()
results_df = load_results()
best_model_name = joblib.load(os.path.join(MODELS_DIR, "best_model_name.pkl"))

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA - Visual Insights", "Model Performance",
     "Predict Customer Churn", "Feature Importance", "Business Summary"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Customer Churn Prediction")
st.sidebar.markdown("**Best Model:** " + best_model_name)
st.sidebar.markdown("**Dataset:** IBM Telco Customer Churn")

# ----------------------------------------------------------------------
# PAGE 1: Overview
# ----------------------------------------------------------------------
if page == "Overview":
    st.title("📊 Customer Churn Prediction & Retention Analytics")
    st.markdown("""
    This dashboard predicts which telecom customers are likely to **churn**
    (cancel their service), and shows the key business factors driving churn.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{df.shape[0]:,}")
    col2.metric("Churned Customers", f"{(df['Churn']=='Yes').sum():,}")
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    avg_tenure = df['tenure'].mean()
    col4.metric("Avg. Tenure (months)", f"{avg_tenure:.1f}")

    st.markdown("---")
    st.subheader("Sample of the Dataset")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Dataset Summary")
    st.write(f"- **Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
    st.write("- **Target column:** `Churn` (Yes / No)")
    st.write("- **Source:** IBM Telco Customer Churn dataset")

# ----------------------------------------------------------------------
# PAGE 2: EDA
# ----------------------------------------------------------------------
elif page == "EDA - Visual Insights":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Visual answers to key business questions about churn.")

    charts = [
        ("01_churn_distribution.png", "Overall churn is about 26.5% of all customers — a meaningful but imbalanced split."),
        ("03_contract_vs_churn.png", "Month-to-month customers churn far more than 1-year or 2-year contract customers."),
        ("04_internet_service_vs_churn.png", "Fiber optic customers have the highest churn rate among internet service types."),
        ("06_tenure_distribution.png", "New customers (low tenure) are much more likely to churn than long-term customers."),
        ("05_monthly_charges_vs_churn.png", "Customers who churn tend to pay higher monthly charges on average."),
        ("08_payment_method_vs_churn.png", "Customers paying via Electronic Check churn at a noticeably higher rate."),
        ("02_gender_vs_churn.png", "Gender has almost no effect on churn — confirms it's not a useful predictor."),
        ("07_correlation_heatmap.png", "Tenure is negatively correlated with churn — longer-tenured customers are more loyal."),
    ]

    for filename, insight in charts:
        path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(path):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(path, use_container_width=True)
            with col2:
                st.markdown(f"**Business Insight:**\n\n{insight}")
            st.markdown("---")

# ----------------------------------------------------------------------
# PAGE 3: Model Performance
# ----------------------------------------------------------------------
elif page == "Model Performance":
    st.title("🤖 Model Performance Comparison")

    st.markdown(f"**Best Model Selected:** `{best_model_name}` (highest F1 Score)")

    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}",
            "F1 Score": "{:.2%}", "ROC-AUC": "{:.2%}"
        }),
        use_container_width=True
    )

    comp_path = os.path.join(IMAGES_DIR, "09_model_comparison.png")
    if os.path.exists(comp_path):
        st.image(comp_path, use_container_width=True)

    st.markdown("### Confusion Matrices")
    cm_cols = st.columns(3)
    cm_files = [
        ("cm_logistic_regression.png", "Logistic Regression"),
        ("cm_decision_tree.png", "Decision Tree"),
        ("cm_random_forest.png", "Random Forest"),
    ]
    for col, (filename, label) in zip(cm_cols, cm_files):
        path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(path):
            col.image(path, caption=label, use_container_width=True)

    st.markdown("""
    ### Why F1 Score for model selection?
    The dataset is imbalanced (~26.5% churners). Accuracy alone can be
    misleading — a model that just predicts "No Churn" for everyone would
    still be ~73% accurate while being completely useless for the business.
    F1 Score balances **Precision** (avoiding wasted retention offers) and
    **Recall** (catching customers who are about to leave).
    """)

# ----------------------------------------------------------------------
# PAGE 4: Live Prediction
# ----------------------------------------------------------------------
elif page == "Predict Customer Churn":
    st.title("🔮 Predict Customer Churn")
    st.markdown("Enter a customer's details below to predict their churn risk.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        result = predict_churn(customer, model, encoders, scaler, feature_columns)

        st.markdown("---")
        if result["prediction"] == "Churn":
            st.error(f"⚠️ **Likely to Churn** — Probability: {result['probability']*100:.1f}%")
            st.markdown("**Recommended Action:** Offer a retention deal, such as a discount "
                        "or contract upgrade incentive, before this customer leaves.")
        else:
            st.success(f"✅ **Not Likely to Churn** — Probability of churn: {result['probability']*100:.1f}%")
            st.markdown("**Recommended Action:** No immediate action needed; continue regular engagement.")

        st.progress(min(result["probability"], 1.0))

# ----------------------------------------------------------------------
# PAGE 5: Feature Importance
# ----------------------------------------------------------------------
elif page == "Feature Importance":
    st.title("📌 Feature Importance")
    st.markdown(f"Which features matter most to the **{best_model_name}** model?")

    fi_path = os.path.join(IMAGES_DIR, "10_feature_importance.png")
    if os.path.exists(fi_path):
        st.image(fi_path, use_container_width=True)

    fi_csv_path = os.path.join(PROCESSED_DIR, "feature_importance.csv")
    if os.path.exists(fi_csv_path):
        fi_df = pd.read_csv(fi_csv_path)
        st.dataframe(fi_df, use_container_width=True)

    st.markdown("""
    ### Business Meaning
    - **Tenure** — newer customers are at the highest risk of churning.
    - **Contract type** — month-to-month customers churn far more than
      customers locked into 1 or 2-year contracts.
    - **Monthly/Total Charges** — customers paying more without added
      long-term commitment are more likely to leave.
    """)

# ----------------------------------------------------------------------
# PAGE 6: Business Summary
# ----------------------------------------------------------------------
elif page == "Business Summary":
    st.title("📈 Business Summary & Recommendations")

    st.markdown("""
    ### Key Findings
    1. **Contract type is the strongest lever.** Month-to-month customers
       churn at a much higher rate than yearly contract customers.
       → Incentivize customers to switch to longer contracts.
    2. **New customers are at the highest risk.** Churn is concentrated in
       the first few months of the relationship.
       → Build a stronger onboarding / early-engagement program.
    3. **Fiber optic and high-paying customers churn more.**
       → Investigate service quality / pricing perception for fiber customers.
    4. **Electronic check payers churn more than automatic payment users.**
       → Encourage auto-pay enrollment (bank transfer / credit card) with
         small incentives, since auto-pay customers are stickier.

    ### Suggested Business Actions
    - Target month-to-month, low-tenure, high-paying customers with
      proactive retention offers (the model's prediction tool can score
      these customers automatically).
    - A/B test contract upgrade incentives (e.g. discount for moving from
      month-to-month to 1-year contract).
    - Monitor churn-risk score for every customer monthly using this model.

    ### Model Used in Production
    """)
    st.info(f"**{best_model_name}** was selected as the production model "
            f"based on F1 Score, which best balances catching churners "
            f"(Recall) against not over-targeting loyal customers (Precision).")
