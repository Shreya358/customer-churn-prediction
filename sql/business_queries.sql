-- ============================================================
-- business_queries.sql
-- ============================================================
-- 18 business-focused SQL queries for the Customer Churn project.
-- Database: MySQL 8.0
-- Table: customer_churn_db.customers
--
-- Each query answers a specific business question. Run schema.sql and
-- load_to_mysql.py first so this table actually has data.
-- ============================================================

USE customer_churn_db;

-- ------------------------------------------------------------
-- Q1. Overall churn rate
-- Business question: What percentage of our customers have churned?
-- ------------------------------------------------------------
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers;


-- ------------------------------------------------------------
-- Q2. Total monthly revenue and revenue lost to churn
-- Business question: How much recurring revenue are we losing to churned customers?
-- ------------------------------------------------------------
SELECT
    ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost_to_churn,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END) * 100.0 / SUM(monthly_charges), 2) AS pct_revenue_at_risk
FROM customers;


-- ------------------------------------------------------------
-- Q3. Average tenure of churned vs retained customers
-- Business question: How long do customers typically stay before leaving?
-- ------------------------------------------------------------
SELECT
    churn,
    ROUND(AVG(tenure), 1) AS avg_tenure_months,
    COUNT(*) AS num_customers
FROM customers
GROUP BY churn;


-- ------------------------------------------------------------
-- Q4. Churn rate by payment method
-- Business question: Which payment method has the highest churn risk?
-- ------------------------------------------------------------
SELECT
    payment_method,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- Q5. Churn rate by contract type
-- Business question: Does contract length reduce churn risk?
-- ------------------------------------------------------------
SELECT
    contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- Q6. Churn rate by internet service type
-- Business question: Are fiber optic customers really churning more?
-- ------------------------------------------------------------
SELECT
    internet_service,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- Q7. High-risk customers: low tenure + month-to-month + high monthly charge
-- Business question: Which exact customers should retention teams call TODAY?
-- ------------------------------------------------------------
SELECT
    customer_id, tenure, contract, monthly_charges, payment_method, churn
FROM customers
WHERE tenure <= 6
  AND contract = 'Month-to-month'
  AND monthly_charges > 70
ORDER BY monthly_charges DESC
LIMIT 20;


-- ------------------------------------------------------------
-- Q8. Customer segmentation by tenure buckets
-- Business question: How is our customer base distributed across lifecycle stages?
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-1 Year'
        WHEN tenure <= 24 THEN '1-2 Years'
        WHEN tenure <= 48 THEN '2-4 Years'
        ELSE '4+ Years'
    END AS tenure_segment,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_segment
ORDER BY tenure_segment;


-- ------------------------------------------------------------
-- Q9. Average monthly charges: churned vs retained
-- Business question: Are churners paying more on average than loyal customers?
-- ------------------------------------------------------------
SELECT
    churn,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2) AS avg_total_charges
FROM customers
GROUP BY churn;


-- ------------------------------------------------------------
-- Q10. Gender distribution and churn rate by gender
-- Business question: Is gender a meaningful churn driver?
-- ------------------------------------------------------------
SELECT
    gender,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY gender;


-- ------------------------------------------------------------
-- Q11. Senior citizens vs non-senior citizens churn comparison
-- Business question: Do senior citizens churn at a different rate?
-- ------------------------------------------------------------
SELECT
    senior_citizen,
    COUNT(*) AS total_customers,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY senior_citizen;


-- ------------------------------------------------------------
-- Q12. Impact of having a partner/dependents on churn
-- Business question: Are customers with families more loyal?
-- ------------------------------------------------------------
SELECT
    partner,
    dependents,
    COUNT(*) AS total_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY partner, dependents
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- Q13. Effect of add-on services (Tech Support) on churn
-- Business question: Does having Tech Support reduce churn?
-- ------------------------------------------------------------
SELECT
    tech_support,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY tech_support
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- Q14. Effect of Online Security add-on on churn
-- Business question: Does Online Security reduce churn risk?
-- ------------------------------------------------------------
SELECT
    online_security,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY online_security
ORDER BY churn_rate_pct DESC;


-- ------------------------------------------------------------
-- Q15. Paperless billing vs churn
-- Business question: Does billing method correlate with churn?
-- ------------------------------------------------------------
SELECT
    paperless_billing,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY paperless_billing;


-- ------------------------------------------------------------
-- Q16. Top 10 highest-paying customers who churned (biggest revenue loss)
-- Business question: Which lost customers hurt revenue the most?
-- ------------------------------------------------------------
SELECT
    customer_id, contract, tenure, monthly_charges, total_charges
FROM customers
WHERE churn = 'Yes'
ORDER BY monthly_charges DESC
LIMIT 10;


-- ------------------------------------------------------------
-- Q17. Monthly charges distribution by contract type (avg, min, max)
-- Business question: How does pricing vary across contract types?
-- ------------------------------------------------------------
SELECT
    contract,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(MIN(monthly_charges), 2) AS min_monthly_charges,
    ROUND(MAX(monthly_charges), 2) AS max_monthly_charges,
    COUNT(*) AS total_customers
FROM customers
GROUP BY contract;


-- ------------------------------------------------------------
-- Q18. Customer value segmentation (combining tenure + spend)
-- Business question: Who are our most valuable AND most at-risk customers?
-- ------------------------------------------------------------
SELECT
    customer_id,
    tenure,
    total_charges,
    monthly_charges,
    contract,
    churn,
    CASE
        WHEN total_charges >= 4000 THEN 'High Value'
        WHEN total_charges >= 1000 THEN 'Medium Value'
        ELSE 'Low Value'
    END AS customer_value_segment
FROM customers
ORDER BY total_charges DESC
LIMIT 20;
