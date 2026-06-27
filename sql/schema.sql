-- ============================================================
-- schema.sql
-- ============================================================
-- Creates the database table that stores cleaned customer churn data.
-- Run this FIRST, before loading any data.
--
-- Database engine: MySQL 8.0
-- ============================================================

CREATE DATABASE IF NOT EXISTS customer_churn_db;
USE customer_churn_db;

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id        VARCHAR(20)    PRIMARY KEY,
    gender              VARCHAR(10),
    senior_citizen      VARCHAR(5),
    partner             VARCHAR(5),
    dependents          VARCHAR(5),
    tenure              INT,
    phone_service       VARCHAR(5),
    multiple_lines      VARCHAR(20),
    internet_service    VARCHAR(20),
    online_security     VARCHAR(25),
    online_backup       VARCHAR(25),
    device_protection   VARCHAR(25),
    tech_support        VARCHAR(25),
    streaming_tv        VARCHAR(25),
    streaming_movies    VARCHAR(25),
    contract            VARCHAR(20),
    paperless_billing   VARCHAR(5),
    payment_method      VARCHAR(30),
    monthly_charges     DECIMAL(8,2),
    total_charges       DECIMAL(10,2),
    churn               VARCHAR(5)
);

-- Helpful indexes for the kinds of business queries we'll run
CREATE INDEX idx_churn ON customers(churn);
CREATE INDEX idx_contract ON customers(contract);
CREATE INDEX idx_internet_service ON customers(internet_service);
CREATE INDEX idx_payment_method ON customers(payment_method);
