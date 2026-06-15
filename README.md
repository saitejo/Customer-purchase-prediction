# Customer-purchase-prediction


A machine learning model that predicts whether an e-commerce 
user will make a purchase, based on their browsing behavior 
and demographic data.

## Overview

Built a complete end-to-end ML pipeline — from raw data 
exploration to final predictions — using Python and scikit-learn. 
The model analyzes behavioral patterns like pages viewed, 
time on site, and past purchases to classify users as 
likely to convert or not.

## Dataset

Anonymized e-commerce user data with the following features:

| Feature | Description |
|---------|-------------|
| Age | User age |
| Income | Estimated annual income |
| City_Tier | City classification (1/2/3) |
| Pages_Viewed | Number of pages viewed per session |
| Products_Viewed | Number of products browsed |
| Time_On_Site | Session duration (minutes) |
| Previous_Purchases | Historical purchase count |
| Discount_Seen | Whether user saw a discount (0/1) |
| Browser_Version | Browser version identifier |
| Campaign_Code | Marketing campaign identifier |

**Target Variable:** `Converted` — 1 (purchased) or 0 (did not purchase)

> Note: Raw dataset not included in this repository.

## ML Pipeline

### 1. Exploratory Data Analysis
- Checked data distributions and class balance
- Identified missing values (~15% in Age, ~10% in Income)
- Found target imbalance: ~69% non-converted, ~31% converted

### 2. Preprocessing
- **Missing values:** Median imputation (fit on train, 
  applied to test — no data leakage)
- **Categorical columns:** Dropped for baseline 
  (Device_Type, Traffic_Source)
- **Feature scaling:** StandardScaler (fit on train, 
  applied to test)

### 3. Model
- **Algorithm:** Logistic Regression
- **Why:** Simple, interpretable, strong baseline 
  for binary classification
- **Parameters:** max_iter=1000, random_state=42

### 4. Evaluation
- **Metric:** F1 Score (accounts for class imbalance)
- **Training Accuracy:** 70.97%

## Key Findings

Most important features (by coefficient magnitude):
