from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    ConfusionMatrixDisplay,
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from xgboost import XGBClassifier
from tensorflow import keras

from src.data import load_raw, clean, make_splits
from src.features import make_scaled_arrays
from src.train_baselines import best_class, best_reg

np.random.seed(42)

REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = Path("models")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load data
# =========================================================
df_raw = load_raw()
df = clean(df_raw)

(
    X_train,
    X_val,
    X_test,
    y_class_train,
    y_class_val,
    y_class_test,
    y_reg_train,
    y_reg_val,
    y_reg_test,
) = make_splits(df)

# NN uses scaled data
_, X_train_scaled, X_val_scaled, X_test_scaled = make_scaled_arrays(
    X_train, X_val, X_test
)

X_val_scaled = X_val_scaled.astype("float32")
X_test_scaled = X_test_scaled.astype("float32")

y_class_val = np.asarray(y_class_val).astype(int)
y_class_test = np.asarray(y_class_test).astype(int)


# =========================================================
# -------- Neural Network: Threshold Optimization ----------
# =========================================================
model_nn = keras.models.load_model(
    MODELS_DIR / "nn_classifier.h5", compile=False
)

y_val_proba_nn = model_nn.predict(X_val_scaled).ravel()
y_test_proba_nn = model_nn.predict(X_test_scaled).ravel()

thresholds = np.linspace(0.05, 0.95, 91)
f1_scores_nn = []

for t in thresholds:
    preds = (y_val_proba_nn >= t).astype(int)
    f1_scores_nn.append(f1_score(y_class_val, preds))

best_idx_nn = int(np.argmax(f1_scores_nn))
best_threshold_nn = thresholds[best_idx_nn]
best_f1_val_nn = f1_scores_nn[best_idx_nn]

# Default vs optimized
y_val_pred_nn_def = (y_val_proba_nn >= 0.5).astype(int)
y_val_pred_nn_opt = (y_val_proba_nn >= best_threshold_nn).astype(int)

y_test_pred_nn_def = (y_test_proba_nn >= 0.5).astype(int)
y_test_pred_nn_opt = (y_test_proba_nn >= best_threshold_nn).astype(int)

acc_val_nn = accuracy_score(y_class_val, y_val_pred_nn_def)
f1_val_nn = f1_score(y_class_val, y_val_pred_nn_def)
auc_val_nn = roc_auc_score(y_class_val, y_val_proba_nn)

acc_val_nn_opt = accuracy_score(y_class_val, y_val_pred_nn_opt)
f1_val_nn_opt = f1_score(y_class_val, y_val_pred_nn_opt)

acc_test_nn = accuracy_score(y_class_test, y_test_pred_nn_def)
f1_test_nn = f1_score(y_class_test, y_test_pred_nn_def)
auc_test_nn = roc_auc_score(y_class_test, y_test_proba_nn)

acc_test_nn_opt = accuracy_score(y_class_test, y_test_pred_nn_opt)
f1_test_nn_opt = f1_score(y_class_test, y_test_pred_nn_opt)


nn_threshold_table = pd.DataFrame([
    ["NN_MLP", 0.5, "val", acc_val_nn, f1_val_nn, auc_val_nn],
    ["NN_MLP", best_threshold_nn, "val", acc_val_nn_opt, f1_val_nn_opt, auc_val_nn],
    ["NN_MLP", 0.5, "test", acc_test_nn, f1_test_nn, auc_test_nn],
    ["NN_MLP", best_threshold_nn, "test", acc_test_nn_opt, f1_test_nn_opt, auc_test_nn],
], columns=["Model", "Threshold", "Split", "Accuracy", "F1", "ROC_AUC"])

nn_threshold_table.to_csv(
    TABLES_DIR / "table3_nn_threshold_optimization.csv",
    index=False,
)


# =========================================================
# -------- XGBoost: Threshold Optimization ----------------
# =========================================================
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=(
        (y_class_train == 0).sum() / (y_class_train == 1).sum()
    ),
    n_jobs=-1,
)

# Train ONLY on training split
xgb_model.fit(X_train, y_class_train)

y_val_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
y_test_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

f1_scores_xgb = []
for t in thresholds:
    preds = (y_val_proba_xgb >= t).astype(int)
    f1_scores_xgb.append(f1_score(y_class_val, preds))

best_idx_xgb = int(np.argmax(f1_scores_xgb))
best_threshold_xgb = thresholds[best_idx_xgb]
best_f1_val_xgb = f1_scores_xgb[best_idx_xgb]

# Default vs optimized
y_val_pred_xgb_def = (y_val_proba_xgb >= 0.5).astype(int)
y_val_pred_xgb_opt = (y_val_proba_xgb >= best_threshold_xgb).astype(int)

y_test_pred_xgb_def = (y_test_proba_xgb >= 0.5).astype(int)
y_test_pred_xgb_opt = (y_test_proba_xgb >= best_threshold_xgb).astype(int)

acc_val_xgb = accuracy_score(y_class_val, y_val_pred_xgb_def)
f1_val_xgb = f1_score(y_class_val, y_val_pred_xgb_def)
auc_val_xgb = roc_auc_score(y_class_val, y_val_proba_xgb)

acc_val_xgb_opt = accuracy_score(y_class_val, y_val_pred_xgb_opt)
f1_val_xgb_opt = f1_score(y_class_val, y_val_pred_xgb_opt)

acc_test_xgb = accuracy_score(y_class_test, y_test_pred_xgb_def)
f1_test_xgb = f1_score(y_class_test, y_test_pred_xgb_def)
auc_test_xgb = roc_auc_score(y_class_test, y_test_proba_xgb)

acc_test_xgb_opt = accuracy_score(y_class_test, y_test_pred_xgb_opt)
f1_test_xgb_opt = f1_score(y_class_test, y_test_pred_xgb_opt)


xgb_threshold_table = pd.DataFrame([
    ["XGBoost", 0.5, "val", acc_val_xgb, f1_val_xgb, auc_val_xgb],
    ["XGBoost", best_threshold_xgb, "val", acc_val_xgb_opt, f1_val_xgb_opt, auc_val_xgb],
    ["XGBoost", 0.5, "test", acc_test_xgb, f1_test_xgb, auc_test_xgb],
    ["XGBoost", best_threshold_xgb, "test", acc_test_xgb_opt, f1_test_xgb_opt, auc_test_xgb],
], columns=["Model", "Threshold", "Split", "Accuracy", "F1", "ROC_AUC"])

xgb_threshold_table.to_csv(
    TABLES_DIR / "table4_xgb_threshold_optimization.csv",
    index=False,
)


# =========================================================
# Confusion matrix (XGBoost, optimized)
# =========================================================
ConfusionMatrixDisplay.from_predictions(
    y_class_test, y_test_pred_xgb_opt, cmap="Blues"
)
plt.title("Confusion Matrix — XGBoost (Test, Optimized Threshold)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "plot6_xgb_confusion_matrix.png", dpi=200)
plt.clf()


# =========================================================
# Console summary
# =========================================================
print("\n=== THRESHOLD OPTIMIZATION SUMMARY ===")
print(f"NN optimal threshold:  {best_threshold_nn:.3f} | val F1: {best_f1_val_nn:.3f}")
print(f"XGB optimal threshold: {best_threshold_xgb:.3f} | val F1: {best_f1_val_xgb:.3f}")
print("Evaluation complete.")