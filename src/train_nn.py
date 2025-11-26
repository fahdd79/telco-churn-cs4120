from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from src.data import load_raw, clean, make_splits
from src.features import make_scaled_arrays

np.random.seed(42)
tf.random.set_seed(42)

REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"
MODELS_DIR = Path("models")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_classification_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_regression_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


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

input_dim = X_train_scaled.shape[1]
batch_size = 32
max_epochs = 50

mlflow.tensorflow.autolog(disable=True)

with mlflow.start_run(run_name="nn_classification"):
    model_clf = build_classification_model(input_dim)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history_clf = model_clf.fit(
        X_train_scaled,
        y_class_train,
        validation_data=(X_val_scaled, y_class_val),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es],
    )

    y_val_proba = model_clf.predict(X_val_scaled).ravel()
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    acc_val = accuracy_score(y_class_val, y_val_pred)
    f1_val = f1_score(y_class_val, y_val_pred)
    auc_val = roc_auc_score(y_class_val, y_val_proba)

    y_test_proba = model_clf.predict(X_test_scaled).ravel()
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    acc_test = accuracy_score(y_class_test, y_test_pred)
    f1_test = f1_score(y_class_test, y_test_pred)
    auc_test = roc_auc_score(y_class_test, y_test_proba)

    mlflow.log_param("task", "classification")
    mlflow.log_param("model_type", "MLP")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("max_epochs", max_epochs)

    mlflow.log_metrics({
        "acc_val": acc_val,
        "f1_val": f1_val,
        "auc_val": auc_val,
        "acc_test": acc_test,
        "f1_test": f1_test,
        "auc_test": auc_test,
    })

    epochs_clf = range(1, len(history_clf.history["loss"]) + 1)
    plt.figure()
    plt.plot(epochs_clf, history_clf.history["loss"], label="train_loss")
    plt.plot(epochs_clf, history_clf.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Learning Curve — Classification NN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot1_nn_classification_learning_curve.png", dpi=200)
    plt.clf()

    model_clf.save(MODELS_DIR / "nn_classifier.h5")

with mlflow.start_run(run_name="nn_regression"):
    model_reg = build_regression_model(input_dim)
    es_reg = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history_reg = model_reg.fit(
        X_train_scaled,
        y_reg_train,
        validation_data=(X_val_scaled, y_reg_val),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[es_reg],
    )

    y_val_pred_reg = model_reg.predict(X_val_scaled).ravel()
    mae_val = mean_absolute_error(y_reg_val, y_val_pred_reg)
    rmse_val = np.sqrt(mean_squared_error(y_reg_val, y_val_pred_reg))

    y_test_pred_reg = model_reg.predict(X_test_scaled).ravel()
    mae_test = mean_absolute_error(y_reg_test, y_test_pred_reg)
    rmse_test = np.sqrt(mean_squared_error(y_reg_test, y_test_pred_reg))

    mlflow.log_param("task", "regression")
    mlflow.log_param("model_type", "MLP")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("max_epochs", max_epochs)

    mlflow.log_metrics({
        "mae_val": mae_val,
        "rmse_val": rmse_val,
        "mae_test": mae_test,
        "rmse_test": rmse_test,
    })

    epochs_reg = range(1, len(history_reg.history["loss"]) + 1)
    plt.figure()
    plt.plot(epochs_reg, history_reg.history["loss"], label="train_loss")
    plt.plot(epochs_reg, history_reg.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Learning Curve — Regression NN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "plot2_nn_regression_learning_curve.png", dpi=200)
    plt.clf()

    model_reg.save(MODELS_DIR / "nn_regressor.h5")

print("NN training complete.")