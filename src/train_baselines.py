from pathlib import Path
from src.data import load_raw, clean, make_splits

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

np.random.seed(42)

REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"
TABLES_DIR = REPORTS_DIR / "tables"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Data
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

print("INPUT FEATURES =", X_train.shape[1])


# =========================================================
# CLASSIFICATION BASELINES (UPDATED)
# =========================================================
class_models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "DecisionTreeClassifier": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),

    # -------- NEW: XGBoost --------
    "XGBoost": XGBClassifier(
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
        n_jobs=-1
    )
}

rows_cls = []
trained_class_models = {}

for name, model in class_models.items():
    with mlflow.start_run(run_name=f"class_{name}", nested=True):

        # -------- FIT (TRAIN ONLY) --------
        model.fit(X_train, y_class_train)

        # -------- VALIDATION --------
        yv = model.predict(X_val)
        acc_v = accuracy_score(y_class_val, yv)
        f1_v = f1_score(y_class_val, yv)

        try:
            yv_proba = model.predict_proba(X_val)[:, 1]
            auc_v = roc_auc_score(y_class_val, yv_proba)
        except Exception:
            auc_v = np.nan

        # -------- TEST --------
        yt = model.predict(X_test)
        acc_t = accuracy_score(y_class_test, yt)
        f1_t = f1_score(y_class_test, yt)

        try:
            yt_proba = model.predict_proba(X_test)[:, 1]
            auc_t = roc_auc_score(y_class_test, yt_proba)
        except Exception:
            auc_t = np.nan

        # -------- LOGGING --------
        mlflow.log_param("task", "classification")
        mlflow.log_param("model", name)

        mlflow.log_metrics({
            "accuracy_val": acc_v,
            "f1_val": f1_v,
            "accuracy_test": acc_t,
            "f1_test": f1_t
        })

        if not np.isnan(auc_v):
            mlflow.log_metric("roc_auc_val", auc_v)
        if not np.isnan(auc_t):
            mlflow.log_metric("roc_auc_test", auc_t)

        rows_cls.append([name, acc_v, f1_v, auc_v, acc_t, f1_t, auc_t])
        trained_class_models[name] = model


# =========================================================
# Save classification table
# =========================================================
table_cls = pd.DataFrame(
    rows_cls,
    columns=[
        "Model",
        "Accuracy_val",
        "F1_val",
        "ROC_AUC_val",
        "Accuracy_test",
        "F1_test",
        "ROC_AUC_test",
    ],
)

table_cls.to_csv(TABLES_DIR / "table1_classification_metrics.csv", index=False)


# =========================================================
# Select best classification model (by F1_val)
# =========================================================
best_class = table_cls.sort_values("F1_val", ascending=False)["Model"].iloc[0]
best_class_pipe = trained_class_models[best_class]

# Refit on TRAIN + VAL (standard practice)
best_class_pipe.fit(
    pd.concat([X_train, X_val]),
    pd.concat([y_class_train, y_class_val])
)

yt_pred = best_class_pipe.predict(X_test)

ConfusionMatrixDisplay.from_predictions(
    y_class_test, yt_pred, cmap="Blues"
)
plt.title(f"Confusion Matrix — {best_class} (Test)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "plot3_confusion_matrix.png", dpi=200)
plt.clf()


# =========================================================
# REGRESSION BASELINES (UNCHANGED)
# =========================================================
reg_models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "DecisionTreeRegressor": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeRegressor(random_state=42))
    ]),
}

rows_reg = []
trained_reg_models = {}

for name, pipe in reg_models.items():
    with mlflow.start_run(run_name=f"reg_{name}", nested=True):
        pipe.fit(X_train, y_reg_train)

        yv = pipe.predict(X_val)
        mae_v = mean_absolute_error(y_reg_val, yv)
        rmse_v = np.sqrt(mean_squared_error(y_reg_val, yv))

        yt = pipe.predict(X_test)
        mae_t = mean_absolute_error(y_reg_test, yt)
        rmse_t = np.sqrt(mean_squared_error(y_reg_test, yt))

        mlflow.log_param("task", "regression")
        mlflow.log_param("model", name)
        mlflow.log_metrics({
            "mae_val": mae_v,
            "rmse_val": rmse_v,
            "mae_test": mae_t,
            "rmse_test": rmse_t
        })

        rows_reg.append([name, mae_v, rmse_v, mae_t, rmse_t])
        trained_reg_models[name] = pipe


table_reg = pd.DataFrame(
    rows_reg,
    columns=["Model", "MAE_val", "RMSE_val", "MAE_test", "RMSE_test"]
)
table_reg.to_csv(TABLES_DIR / "table2_regression_metrics.csv", index=False)

best_reg = table_reg.sort_values("RMSE_val", ascending=True)["Model"].iloc[0]

print("Best classification baseline:", best_class)
print("Best regression baseline:", best_reg)