from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    ConfusionMatrixDisplay,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tensorflow import keras

from src.data import load_raw, clean, make_splits
from src.features import make_scaled_arrays

np.random.seed(42)

REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = Path("models")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_classical_classification_results(
    X_train, X_val, X_test, y_train, y_val, y_test
):
    models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "DecisionTreeClassifier": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(random_state=42)),
            ]
        ),
    }

    rows = []
    trained = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)

        yv = pipe.predict(X_val)
        acc_v = accuracy_score(y_val, yv)
        f1_v = f1_score(y_val, yv)

        try:
            yv_proba = pipe.predict_proba(X_val)[:, 1]
            auc_v = roc_auc_score(y_val, yv_proba)
        except Exception:
            auc_v = np.nan

        yt = pipe.predict(X_test)
        acc_t = accuracy_score(y_test, yt)
        f1_t = f1_score(y_test, yt)

        try:
            yt_proba = pipe.predict_proba(X_test)[:, 1]
            auc_t = roc_auc_score(y_test, yt_proba)
        except Exception:
            auc_t = np.nan

        rows.append(
            [name, acc_v, f1_v, auc_v, acc_t, f1_t, auc_t]
        )
        trained[name] = pipe

    table = pd.DataFrame(
        rows,
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
    best_name = table.sort_values("F1_val", ascending=False)["Model"].iloc[0]
    best_pipe = trained[best_name]

    return best_name, best_pipe, table


def get_classical_regression_results(
    X_train, X_val, X_test, y_train, y_val, y_test
):
    models = {
        "LinearRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "DecisionTreeRegressor": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeRegressor(random_state=42)),
            ]
        ),
    }

    rows = []
    trained = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)

        yv = pipe.predict(X_val)
        mae_v = mean_absolute_error(y_val, yv)
        rmse_v = np.sqrt(mean_squared_error(y_val, yv))

        yt = pipe.predict(X_test)
        mae_t = mean_absolute_error(y_test, yt)
        rmse_t = np.sqrt(mean_squared_error(y_test, yt))

        rows.append([name, mae_v, rmse_v, mae_t, rmse_t])
        trained[name] = pipe

    table = pd.DataFrame(
        rows, columns=["Model", "MAE_val", "RMSE_val", "MAE_test", "RMSE_test"]
    )
    best_name = table.sort_values("RMSE_val", ascending=True)["Model"].iloc[0]
    best_pipe = trained[best_name]

    return best_name, best_pipe, table


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

scaler, X_train_scaled, X_val_scaled, X_test_scaled = make_scaled_arrays(
    X_train, X_val, X_test
)

X_train_scaled = X_train_scaled.astype("float32")
X_val_scaled = X_val_scaled.astype("float32")
X_test_scaled = X_test_scaled.astype("float32")

y_class_train = np.asarray(y_class_train).astype("float32")
y_class_val = np.asarray(y_class_val).astype("float32")
y_class_test = np.asarray(y_class_test).astype("float32")

y_reg_train = np.asarray(y_reg_train).astype("float32")
y_reg_val = np.asarray(y_reg_val).astype("float32")
y_reg_test = np.asarray(y_reg_test).astype("float32")

best_classical_cls_name, best_classical_cls_pipe, table_cls = get_classical_classification_results(
    X_train, X_val, X_test, y_class_train, y_class_val, y_class_test
)

best_classical_reg_name, best_classical_reg_pipe, table_reg = get_classical_regression_results(
    X_train, X_val, X_test, y_reg_train, y_reg_val, y_reg_test
)

nn_clf_path = MODELS_DIR / "nn_classifier.h5"
nn_reg_path = MODELS_DIR / "nn_regressor.h5"

model_clf = keras.models.load_model(nn_clf_path, compile=False)
model_reg = keras.models.load_model(nn_reg_path, compile=False)

y_val_proba_nn = model_clf.predict(X_val_scaled).ravel()
y_val_pred_nn = (y_val_proba_nn >= 0.5).astype(int)
acc_val_nn = accuracy_score(y_class_val, y_val_pred_nn)
f1_val_nn = f1_score(y_class_val, y_val_pred_nn)
auc_val_nn = roc_auc_score(y_class_val, y_val_proba_nn)

y_test_proba_nn = model_clf.predict(X_test_scaled).ravel()
y_test_pred_nn = (y_test_proba_nn >= 0.5).astype(int)
acc_test_nn = accuracy_score(y_class_test, y_test_pred_nn)
f1_test_nn = f1_score(y_class_test, y_test_pred_nn)
auc_test_nn = roc_auc_score(y_class_test, y_test_proba_nn)

y_val_pred_reg_nn = model_reg.predict(X_val_scaled).ravel()
mae_val_nn = mean_absolute_error(y_reg_val, y_val_pred_reg_nn)
rmse_val_nn = np.sqrt(mean_squared_error(y_reg_val, y_val_pred_reg_nn))

y_test_pred_reg_nn = model_reg.predict(X_test_scaled).ravel()
mae_test_nn = mean_absolute_error(y_reg_test, y_test_pred_reg_nn)
rmse_test_nn = np.sqrt(mean_squared_error(y_reg_test, y_test_pred_reg_nn))

row_classical_cls = table_cls[table_cls["Model"] == best_classical_cls_name]
row_nn_cls = pd.DataFrame(
    [
        [
            "NN_MLP",
            acc_val_nn,
            f1_val_nn,
            auc_val_nn,
            acc_test_nn,
            f1_test_nn,
            auc_test_nn,
        ]
    ],
    columns=row_classical_cls.columns,
)

final_cls_table = pd.concat(
    [row_classical_cls.reset_index(drop=True), row_nn_cls], ignore_index=True
)
final_cls_table.to_csv(
    TABLES_DIR / "table1_classification_metrics.csv", index=False
)

row_classical_reg = table_reg[table_reg["Model"] == best_classical_reg_name]
row_nn_reg = pd.DataFrame(
    [["NN_MLP", mae_val_nn, rmse_val_nn, mae_test_nn, rmse_test_nn]],
    columns=row_classical_reg.columns,
)

final_reg_table = pd.concat(
    [row_classical_reg.reset_index(drop=True), row_nn_reg], ignore_index=True
)
final_reg_table.to_csv(
    TABLES_DIR / "table2_regression_metrics.csv", index=False
)

ConfusionMatrixDisplay.from_predictions(
    y_class_test, y_test_pred_nn, cmap="Blues"
)
plt.title("Confusion Matrix — NN (Test)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "plot3_confusion_matrix.png", dpi=200)
plt.clf()

residuals_nn = y_reg_test - y_test_pred_reg_nn
plt.scatter(y_test_pred_reg_nn, residuals_nn, s=16)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel("Predicted MonthlyCharges (NN)")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted — NN (Test)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "plot4_residuals_vs_predicted.png", dpi=200)
plt.clf()

if best_classical_cls_name == "LogisticRegression":
    coefs = np.abs(
        best_classical_cls_pipe.named_steps["model"].coef_[0]
    )
else:
    coefs = best_classical_cls_pipe.named_steps[
        "model"
    ].feature_importances_

feature_names = X_train.columns
indices = np.argsort(coefs)[::-1]
top_k = min(10, len(indices))
top_idx = indices[:top_k]

plt.barh(
    range(top_k),
    coefs[top_idx][::-1],
)
plt.yticks(
    range(top_k),
    [feature_names[i] for i in top_idx][::-1],
)
plt.xlabel("Importance")
plt.title(f"Feature Importance — {best_classical_cls_name}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "plot5_feature_importance.png", dpi=200)
plt.clf()

print("Evaluation complete.")