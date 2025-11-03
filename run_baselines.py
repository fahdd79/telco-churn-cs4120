import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import mlflow
import mlflow.sklearn

Path("plots").mkdir(exist_ok=True)
Path("tables").mkdir(exist_ok=True)

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})

y_class = df["ChurnFlag"]
y_reg = df["MonthlyCharges"]
X_raw = df.drop(columns=["Churn", "ChurnFlag", "MonthlyCharges", "customerID"])
X = pd.get_dummies(X_raw, drop_first=True)

print("The data set contains " + str(X.shape[0]) + " training examples of " + str(X.shape[1]) + " dimensions each.")

RANDOM_SEED = 42
X_trainval, X_test, y_class_trainval, y_class_test, y_reg_trainval, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.15, random_state=RANDOM_SEED, stratify=y_class
)
X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
    X_trainval, y_class_trainval, y_reg_trainval, test_size=0.1765, random_state=RANDOM_SEED, stratify=y_class_trainval
)

print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

class_models = {
    "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
    "DecisionTreeClassifier": Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeClassifier(random_state=RANDOM_SEED))]),
}

rows_cls = []
for name, pipe in class_models.items():
    with mlflow.start_run(run_name=f"class_{name}", nested=True):
        pipe.fit(X_train, y_class_train)
        yv = pipe.predict(X_val)
        acc_v = accuracy_score(y_class_val, yv)
        f1_v = f1_score(y_class_val, yv)
        try:
            yv_proba = pipe.predict_proba(X_val)[:, 1]
            auc_v = roc_auc_score(y_class_val, yv_proba)
        except Exception:
            auc_v = np.nan
        yt = pipe.predict(X_test)
        acc_t = accuracy_score(y_class_test, yt)
        f1_t = f1_score(y_class_test, yt)
        try:
            yt_proba = pipe.predict_proba(X_test)[:, 1]
            auc_t = roc_auc_score(y_class_test, yt_proba)
        except Exception:
            auc_t = np.nan
        mlflow.log_param("task", "classification")
        mlflow.log_param("model", name)
        mlflow.log_metrics({"accuracy_val": acc_v, "f1_val": f1_v})
        if not np.isnan(auc_v): mlflow.log_metric("roc_auc_val", auc_v)
        mlflow.log_metrics({"accuracy_test": acc_t, "f1_test": f1_t})
        if not np.isnan(auc_t): mlflow.log_metric("roc_auc_test", auc_t)
        rows_cls.append([name, acc_v, f1_v, auc_v, acc_t, f1_t, auc_t])

table_cls = pd.DataFrame(rows_cls, columns=["Model","Accuracy_val","F1_val","ROC_AUC_val","Accuracy_test","F1_test","ROC_AUC_test"])
table_cls.to_csv("tables/table1_classification_metrics.csv", index=False)

best_class = table_cls.sort_values("F1_val", ascending=False)["Model"].iloc[0]
best_class_pipe = class_models[best_class]
best_class_pipe.fit(pd.concat([X_train, X_val]), pd.concat([y_class_train, y_class_val]))
yt_pred = best_class_pipe.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_class_test, yt_pred, cmap="Blues")
plt.title(f"Confusion Matrix — {best_class} (Test)")
plt.tight_layout()
plt.savefig("plots/plot3_confusion_matrix.png", dpi=200)
plt.clf()

reg_models = {
    "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "DecisionTreeRegressor": Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeRegressor(random_state=RANDOM_SEED))]),
}

rows_reg = []
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
        mlflow.log_metrics({"mae_val": mae_v, "rmse_val": rmse_v, "mae_test": mae_t, "rmse_test": rmse_t})
        rows_reg.append([name, mae_v, rmse_v, mae_t, rmse_t])

table_reg = pd.DataFrame(rows_reg, columns=["Model","MAE_val","RMSE_val","MAE_test","RMSE_test"])
table_reg.to_csv("tables/table2_regression_metrics.csv", index=False)

best_reg = table_reg.sort_values("RMSE_val", ascending=True)["Model"].iloc[0]
best_reg_pipe = reg_models[best_reg]
best_reg_pipe.fit(pd.concat([X_train, X_val]), pd.concat([y_reg_train, y_reg_val]))
y_pred_test = best_reg_pipe.predict(X_test)
residuals = y_reg_test - y_pred_test
plt.scatter(y_pred_test, residuals, s=16)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel("Predicted MonthlyCharges")
plt.ylabel("Residuals")
plt.title(f"Residuals vs Predicted — {best_reg} (Test)")
plt.tight_layout()
plt.savefig("plots/plot4_residuals_vs_predicted.png", dpi=200)
plt.clf()

print("Best classification baseline:", best_class)
print("Best regression baseline:", best_reg)
print("Saved plots and tables.")