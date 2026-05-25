import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

DATA_DIR = "datasets"
INPUT_PREFIX = "churn"
OUTPUT_PREFIX = "churn_preprocessed"
TARGET_COL = "Churn"

def _encode_binary_labels(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in ("U", "S", "O"):
        mapped = []
        for value in y_arr:
            key = str(value).strip().lower()
            if key in ("yes", "true", "1"):
                mapped.append(1)
            elif key in ("no", "false", "0"):
                mapped.append(0)
            else:
                raise ValueError(f"Unsupported label value: {value}")
        return np.asarray(mapped, dtype=np.int64)
    return y_arr.astype(np.int64)


def load_split_datasets(
    data_dir: str = "datasets",
    prefix: str = "churn",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_dir)

    train_path = os.path.join(data_path, f"{prefix}_train.csv")
    val_path = os.path.join(data_path, f"{prefix}_val.csv")
    test_path = os.path.join(data_path, f"{prefix}_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing file: {val_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def _coerce_numeric_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
    return df


def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Churn",
    scaler: Optional[RobustScaler] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[RobustScaler],
    Iterable[str],
]:
    train_df = _coerce_numeric_columns(train_df, target_col)
    val_df = _coerce_numeric_columns(val_df, target_col)
    test_df = _coerce_numeric_columns(test_df, target_col)

    y_train = _encode_binary_labels(train_df[target_col].values)
    X_train = train_df.drop(columns=[target_col])

    y_val = _encode_binary_labels(val_df[target_col].values)
    X_val = val_df.drop(columns=[target_col])

    y_test = _encode_binary_labels(test_df[target_col].values)
    X_test = test_df.drop(columns=[target_col])

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    medians = X_train[numeric_cols].median()
    X_train[numeric_cols] = X_train[numeric_cols].fillna(medians)
    X_val[numeric_cols] = X_val[numeric_cols].fillna(medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(medians)

    X_train_enc = pd.get_dummies(X_train, drop_first=False)
    X_val_enc = pd.get_dummies(X_val, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, drop_first=False)

    X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    scaler = RobustScaler()
    scaler.fit(X_train_enc[numeric_cols])

    X_train_enc[numeric_cols] = scaler.transform(X_train_enc[numeric_cols])
    X_val_enc[numeric_cols] = scaler.transform(X_val_enc[numeric_cols])
    X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])

    return (
        X_train_enc.values.astype(np.float32),
        y_train,
        X_val_enc.values.astype(np.float32),
        y_val,
        X_test_enc.values.astype(np.float32),
        y_test,
        scaler,
        X_train_enc.columns,
    )


def load_and_prepare_datasets(
    data_dir: str = "datasets",
    prefix: str = "churn",
    target_col: str = "Churn",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    RobustScaler,
]:
    train_df, val_df, test_df = load_split_datasets(data_dir=data_dir, prefix=prefix)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, _ = preprocess_data(
        train_df, val_df=val_df, test_df=test_df, target_col=target_col
    )
    train_df_prepared = pd.DataFrame(X_train, columns=train_df.drop(columns=[target_col]).columns)
    train_df_prepared[target_col] = y_train

    val_df_prepared = pd.DataFrame(X_val, columns=val_df.drop(columns=[target_col]).columns)
    val_df_prepared[target_col] = y_val

    test_df_prepared = pd.DataFrame(X_test, columns=test_df.drop(columns=[target_col]).columns)
    test_df_prepared[target_col] = y_test
    return (
        train_df_prepared,
        val_df_prepared,
        test_df_prepared,
        scaler,
    )
