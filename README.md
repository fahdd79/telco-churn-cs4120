# Telco Customer Churn — CS-4120 Final Project

## Overview
This project analyzes the IBM Telco Customer Churn dataset to address two predictive tasks:

1. **Classification:** Predict whether a customer will churn (Yes/No).  
2. **Regression:** Predict the customer’s **MonthlyCharges** as a continuous value.

The goal is to understand customer behavior, reduce churn, and identify key factors driving monthly billing.
## Dataset

**Source:** IBM Telco Customer Churn (public Kaggle dataset)  
**Rows:** 7,043  
**Columns after cleaning:** 22  
**Targets:**  
- **ChurnFlag** (0 = No, 1 = Yes) for classification  
- **MonthlyCharges** for regression  

**Missing values:**  
- `TotalCharges` had ~0.16% missing → converted to numeric and dropped  

**Class distribution:**  
- No churn: ~73%  
- Yes churn: ~27%  

**Key numeric features:**  
- tenure  
- MonthlyCharges  
- TotalCharges  


## Methods Overview

### Classical machine-learning baselines
We trained two baseline models for each task using scikit-learn:

**Classification**
- Logistic Regression  
- Decision Tree Classifier  

**Regression**
- Linear Regression  
- Decision Tree Regressor  

All classical models used:
- StandardScaler (inside a Pipeline)
- The same train/validation/test split
- MLflow logging for metrics and parameters

### Neural Network (MLP)
We implemented a fully connected feed-forward neural network in TensorFlow/Keras for both tasks.

**Architecture (shared idea for both models)**
- Input layer of size 29  
- 2 hidden layers (ReLU activations)  
- Dropout for regularization  
- Output layer:  
  - Sigmoid for classification  
  - Linear for regression  
- Optimizer: Adam  
- Loss: binary cross-entropy (classification), MSE (regression)

**Justification**
An MLP (multi-layer perceptron) is appropriate because:
- The dataset is **tabular, non-sequential**  
- CNNs/RNNs do not apply to this structure  
- MLPs can model nonlinear relationships while staying lightweight  
- Dropout improves generalization  

All NNs were trained using the same train/val/test split and logged to MLflow.



## Project Structure
```md
## Project Structure

project/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├├── src/
│   ├── data.py              # loading, cleaning, splitting
│   ├── features.py          # preprocessing utilities
│   ├── train_baselines.py   # classical ML models
│   ├── train_nn.py          # neural network training
│   └── evaluate.py          # evaluation and comparison
│
├── reports/
│   ├── plots/               # generated visualizations
│   └── tables/              # metrics tables
│
├── models/                  # saved neural network models
├── mlruns/                  # MLflow tracking data
├── notebooks/
│   └── EDA.py               # exploratory data analysis
├── requirements.txt
├── LICENSE
└── README.md
```
## 3. Setup Instructions

Follow these steps to set up the project environment and reproduce all results.

### 3.1. Clone the Repository
```bash
git clone <your-repo-url>
cd telco-churn-cs4120
```

### 3.2. Install Dependencies
Make sure you are using **Python 3.11**, required for `tensorflow-macos`.

```bash
pip install -r requirements.txt
```

### 3.3. Dataset Location
Place the dataset here:

```
project_root/
└── data/
    └── Telco-Customer-Churn.csv
```

If the file is already present, no action is needed.

### 3.4. Directory Setup Notes
The following folders are created automatically when you run training scripts:

- `reports/plots/`  
- `reports/tables/`  
- `models/`  
- `mlruns/`  

You do not need to create these manually.

---
## 4. How to Run the Project

This project includes three main executable scripts:  
1. **train_baselines.py** → trains classical ML models  
2. **train_nn.py** → trains neural networks for both tasks  
3. **evaluate.py** → generates all final figures, tables, and comparisons  

Make sure your working directory is the project root before running any script.

---

### 4.1 Run Classical Baseline Models
This trains Logistic Regression, Decision Tree (classification), Linear Regression, and Decision Tree Regressor.

```bash
python src/train_baselines.py
```

Outputs saved to:
```
reports/tables/table1_classification_metrics.csv
reports/tables/table2_regression_metrics.csv
reports/plots/plot3_confusion_matrix.png
reports/plots/plot4_residuals_vs_predicted.png
```

---

### 4.2 Run Neural Networks (MLP)
This trains one neural network for classification and one for regression.

```bash
python src/train_nn.py
```

Outputs saved to:
```
models/nn_classification.h5
models/nn_regression.h5
reports/plots/plot1_nn_classification_learning_curve.png
reports/plots/plot2_nn_regression_learning_curve.png
```

---

### 4.3 Generate All Final Plots and Comparisons
This script loads trained models and produces:

- Feature importance  
- Confusion matrix (NN)  
- Residuals (NN)  
- Tables comparing best baseline vs best NN  

Run:

```bash
python src/evaluate.py
```

Outputs saved to:
```
reports/plots/
reports/tables/
```

---

### 4.4 Optional: Run EDA Script
If you want to reproduce exploratory plots:

```bash
python notebooks/EDA.py
```

Outputs saved to:
```
reports/plots/plot1_churn_distribution.png
reports/plots/plot2_corr_heatmap.png
```

---
## 5. Modeling Summary

This project implements two types of predictive models for both classification and regression tasks:  
**classical machine-learning baselines** and a **fully connected neural network (MLP)**.

---

### 5.1 Classical Models

**Classification Baselines**
- **Logistic Regression**  
- **Decision Tree Classifier**

**Regression Baselines**
- **Linear Regression**  
- **Decision Tree Regressor**

Each baseline model was implemented using a scikit-learn `Pipeline` and evaluated using:
- Accuracy, F1-score, ROC-AUC (classification)  
- MAE, RMSE (regression)

All classical models used the **same train/validation/test split** for fairness.

---

### 5.2 Neural Network Models (MLP)

We implemented two neural networks in TensorFlow/Keras:
- One for **classification** (predicting churn)
- One for **regression** (predicting MonthlyCharges)

**Shared Architecture**
- Input: 29 features  
- Hidden Layer 1: Dense(128, ReLU)  
- Batch Normalization + Dropout(0.3)  
- Hidden Layer 2: Dense(64, ReLU)  
- Batch Normalization + Dropout(0.3)  
- Output Layer:  
  - Sigmoid (classification)  
  - Linear (regression)

**Trainable parameters:**  
Both networks have **12,161 trainable parameters**:

```
(29 × 128 + 128) + (128 × 64 + 64) + (64 × 1 + 1)
```

**Why an MLP?**
- Dataset is **tabular**, not spatial or sequential   
- MLPs handle one-hot categorical variables well  
- Dropout + batch norm help prevent overfitting  

All neural networks used:
- Adam optimizer (lr = 0.001)  
- Early stopping  
- Same data splits as classical models  

---
## 6. Results Summary

This section summarizes the performance of all classical and neural models on the **held-out test set**.

---

### 6.1 Classification Results (Churn Prediction)

**Best Classical Model: Logistic Regression**  
- **Accuracy:** ~0.79  
- **F1-score:** ~0.585  
- **ROC-AUC:** ~0.83  

**Neural Network (MLP) – Classification**  
- **Accuracy:** ~0.79  
- **F1-score:** ~0.57  
- **ROC-AUC:** ~0.82  

**Interpretation**
- Logistic Regression slightly outperforms the neural network on F1-score.  
- Both achieve very similar ROC-AUC, indicating both models separate churn vs non-churn similarly well.  
- Given the dataset is **tabular** with mostly linear relationships, this result is expected.

---

### 6.2 Regression Results (Predicting MonthlyCharges)

**Best Classical Model: Linear Regression**  
- **MAE:** ~0.76  
- **RMSE:** ~1.01  

**Neural Network (MLP) – Regression**  
- **MAE:** ~1.04  
- **RMSE:** ~1.33  

**Interpretation**
- Linear Regression significantly outperforms the neural network on both error metrics.  
- Error increases for customers with higher charges due to mild **heteroscedasticity** (larger variance at high values).  
- Regression performance suggests a mostly linear trend between features and MonthlyCharges.

---

### 6.3 Feature Importance (Classification)

Using Logistic Regression coefficients, we observe:

- **Tenure** and **TotalCharges** reduce churn probability.  
- **Month-to-month contract**, **electronic check**, and **additional internet services** increase churn probability.  

These insights align with business expectations: long-term customers are more stable, while month-to-month customers are more likely to churn.

---
## 7. Reproducibility and MLflow Tracking

Reproducibility is a key component of this project. Every experiment, model, and artifact is logged in a way that allows results to be regenerated exactly as presented.

---

### 7.1 Deterministic Train/Validation/Test Split
All experiments use the same fixed split:

- **70% training**
- **15% validation**
- **15% test**
- `random_state = 42` ensures identical splits on every run.

This guarantees fair comparison between classical models and neural networks.

---

### 7.2 No Data Leakage
All preprocessing steps avoid leakage by design:

- StandardScaler is **fit only on training data**  
- Validation and test data are transformed using the training scaler  
- Splitting occurs **before** any normalization or modeling

This ensures that no information from unseen data influences training.

---

### 7.3 MLflow Tracking
All runs log:

- Model type (baseline or NN)  
- Hyperparameters  
- Train/val/test metrics  
- Saved artifacts (plots, tables, confusion matrices, residual plots)

The MLflow tracking directory is stored in:

```
mlruns/
```

To inspect runs locally:

```bash
mlflow ui
```

Then open:

```
http://127.0.0.1:5000
```

---
## 8. Generated Outputs

All plots, tables, and trained models are automatically saved when you run the training and evaluation scripts.

---

### 8.1 Plots (saved in `reports/plots/`)

**Exploratory Data Analysis**
- `plot1_churn_distribution.png` — Class balance of churn vs. non-churn  
- `plot2_corr_heatmap.png` — Correlation heatmap with numerical annotations  

**Baseline Models**
- `plot3_confusion_matrix.png` — Confusion matrix for best classification baseline  
- `plot4_residuals_vs_predicted.png` — Residual plot for best regression baseline  
- `plot5_feature_importance.png` — Coefficient-based importance from Logistic Regression  

**Neural Networks**
- `plot1_nn_classification_learning_curve.png` — Training/validation loss (classification NN)  
- `plot2_nn_regression_learning_curve.png` — Training/validation loss (regression NN)  

---

### 8.2 Tables (saved in `reports/tables/`)

**Classification**
- `table1_classification_metrics.csv`  
  - Accuracy, F1-score, ROC-AUC for Logistic Regression and Decision Tree  

**Regression**
- `table2_regression_metrics.csv`  
  - MAE and RMSE for Linear Regression and Decision Tree Regressor  

Tables are formatted cleanly and rounded to two decimal places for readability.

---

### 8.3 Trained Models (saved in `models/`)

- `nn_classification.keras`  
- `nn_regression.keras`  

These models can be reloaded using:

```python
from tensorflow import keras
model = keras.models.load_model("models/nn_classification.keras")
```

---
