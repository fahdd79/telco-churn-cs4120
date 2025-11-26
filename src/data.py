from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def load_raw(path: str = "data/Telco-Customer-Churn.csv") -> pd.DataFrame:
    return pd.read_csv(path)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df

def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.1765,
    random_seed: int = RANDOM_SEED,
):
    y_class = df["ChurnFlag"]
    y_reg = df["MonthlyCharges"]

    X_raw = df.drop(columns=["Churn", "ChurnFlag", "MonthlyCharges", "customerID"])
    X = pd.get_dummies(X_raw, drop_first=True)

    X_trainval, X_test, y_class_trainval, y_class_test, y_reg_trainval, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=test_size, random_state=random_seed, stratify=y_class
    )

    X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
        X_trainval,
        y_class_trainval,
        y_reg_trainval,
        test_size=val_size,
        random_state=random_seed,
        stratify=y_class_trainval,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_class_train,
        y_class_val,
        y_class_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
    )