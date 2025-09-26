# Telco Customer Churn — CS-4120 Project Proposal

## Project Overview
This project analyzes the **IBM Telco Customer Churn dataset** to address two predictive tasks:
- **Classification:** Predict whether a customer will churn (`Yes/No`).
- **Regression:** Predict a customer's **MonthlyCharges** as a continuous variable.

The motivation is to help telecom companies improve customer retention strategies and optimize billing plans.

---

## Dataset
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Rows:** 7,043  
- **Columns:** 21  
- **Targets:**  
  - Classification → `Churn` (Yes/No)  
  - Regression → `MonthlyCharges`  
- **Missing values:** ~0.16% in `TotalCharges`  
- **Class distribution:** Yes ≈ 26.5%, No ≈ 73.5%  
- **License:** Open (Kaggle dataset for academic use)

---

## Planned Metrics
- **Classification:** Accuracy, F1-score (ROC–AUC will also be noted if allowed)  
- **Regression:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

---

## Baselines
At least two models per task:
- **Classification:** Logistic Regression, Decision Tree Classifier  
- **Regression:** Linear Regression, Decision Tree Regressor  

---

## Reproducibility
- Dependencies pinned in `requirements.txt`  
- Experiments tracked using **MLflow** (parameters, metrics, artifacts)  
- Repository structure will include: