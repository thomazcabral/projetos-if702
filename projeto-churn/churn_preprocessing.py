import os
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

DATA_DIR = "datasets"
INPUT_PREFIX = "churn"
OUTPUT_PREFIX = "churn_preprocessed"
TARGET_COL = "Churn"

def _encode_binary_labels(
    y: np.ndarray,
    positive_value: Optional[object] = None,
) -> np.ndarray:
    y_arr = np.asarray(y)
    uniques = pd.Series(y_arr).dropna().unique().tolist()

    if positive_value is None:
        if len(uniques) <= 1:
            positive_value = uniques[0] if uniques else 1
        else:
            lower = [str(v).strip().lower() for v in uniques]
            if "yes" in lower:
                positive_value = uniques[lower.index("yes")]
            elif "true" in lower:
                positive_value = uniques[lower.index("true")]
            elif "1" in lower:
                positive_value = uniques[lower.index("1")]
            elif "no" in lower:
                positive_value = uniques[1 - lower.index("no")]
            elif "false" in lower:
                positive_value = uniques[1 - lower.index("false")]
            elif "0" in lower:
                positive_value = uniques[1 - lower.index("0")]
            else:
                positive_value = sorted(uniques, key=lambda x: str(x))[1]

    encoded = (y_arr == positive_value).astype(np.int64)
    return encoded, positive_value


def _encode_multi_labels(
    y: np.ndarray,
    mapping: Optional[Dict[int, Any]] = None,
) -> np.ndarray:
    series = pd.Series(y)
    if series.isna().any():
        series = series.fillna("__MISSING__")

    if mapping is None:
        uniques = series.unique().tolist()
        uniques_sorted = sorted(uniques, key=lambda x: str(x))
        mapping = {idx: value for idx, value in enumerate(uniques_sorted)}

    inverse = {value: idx for idx, value in mapping.items()}
    encoded = series.map(inverse)
    if encoded.isna().any():
        unknown = series[encoded.isna()].unique().tolist()
        raise ValueError(f"Unknown categories found: {unknown}")

    encoded_arr = encoded.astype(np.int64).values
    return encoded_arr, mapping


def load_split_datasets(
    data_dir: str = "datasets",
    prefix: str = "churn",
    train_suffix: str = "_train.csv",
    val_suffix: str = "_val.csv",
    test_suffix: str = "_test.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_dir)

    train_path = os.path.join(data_path, f"{prefix}{train_suffix}")
    val_path = os.path.join(data_path, f"{prefix}{val_suffix}")
    test_path = os.path.join(data_path, f"{prefix}{test_suffix}")

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


def get_X_y_from_split(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = train_df.drop(columns=[target_col])
    y_train = np.asarray(train_df[target_col], dtype=np.int64)
    X_val = val_df.drop(columns=[target_col])
    y_val = np.asarray(val_df[target_col], dtype=np.int64)
    X_test = test_df.drop(columns=[target_col])
    y_test = np.asarray(test_df[target_col], dtype=np.int64)
    return (X_train.values.astype(np.float32), y_train, X_val.values.astype(np.float32), y_val, X_test.values.astype(np.float32), y_test)


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

def encode_categorical_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[int, Any]]]:
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
    binary_cols = []
    multi_cols = []

    for col in cat_cols:
        uniques = pd.Series(X_train[col].dropna().unique()).tolist()
        if len(uniques) == 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)

    ### Binarização de colunas categóricas com 2 categorias (Vira apenas 1 coluna binária)
    X_train_binary = pd.DataFrame(index=X_train.index)
    X_val_binary = pd.DataFrame(index=X_val.index)
    X_test_binary = pd.DataFrame(index=X_test.index)

    for col in binary_cols:
        train_encoded, pos_value = _encode_binary_labels(
            X_train[col].values,
        )
        suffix = str(pos_value).strip().replace(" ", "_")
        col_name = f"{col}_{suffix}"
        X_train_binary[col_name] = train_encoded
        X_val_binary[col_name] = _encode_binary_labels(
            X_val[col].values,
            positive_value=pos_value,
        )
        X_test_binary[col_name] = _encode_binary_labels(
            X_test[col].values,
            positive_value=pos_value,
        )

    ### Codificação de colunas categóricas com mais de 2 categorias (Vira uma coluna numérica com os índices das categorias)
    X_train_multi = pd.DataFrame(index=X_train.index)
    X_val_multi = pd.DataFrame(index=X_val.index)
    X_test_multi = pd.DataFrame(index=X_test.index)
    multi_mappings: Dict[str, Dict[int, Any]] = {}

    for col in multi_cols:
        train_encoded, mapping = _encode_multi_labels(
            X_train[col].values,
        )
        multi_mappings[col] = mapping
        X_train_multi[col] = train_encoded
        X_val_multi[col] = _encode_multi_labels(
            X_val[col].values,
            mapping=mapping,
        )
        X_test_multi[col] = _encode_multi_labels(
            X_test[col].values,
            mapping=mapping,
        )

    X_train_encoded = pd.concat([X_train.drop(columns=cat_cols), X_train_binary, X_train_multi], axis=1)
    X_val_encoded = pd.concat([X_val.drop(columns=cat_cols), X_val_binary, X_val_multi], axis=1)
    X_test_encoded = pd.concat([X_test.drop(columns=cat_cols), X_test_binary, X_test_multi], axis=1)
    return X_train_encoded, X_val_encoded, X_test_encoded, multi_mappings


def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Churn",
    scaler: Optional[RobustScaler] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Iterable[str],
    Dict[str, Dict[int, Any]],
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

    X_train_enc, X_val_enc, X_test_enc, multi_mappings = encode_categorical_features(X_train, X_val, X_test)

    scaler = RobustScaler()
    scaler.fit(X_train_enc[numeric_cols])

    X_train_enc[numeric_cols] = scaler.transform(X_train_enc[numeric_cols])
    X_val_enc[numeric_cols] = scaler.transform(X_val_enc[numeric_cols])
    X_test_enc[numeric_cols] = scaler.transform(X_test_enc[numeric_cols])

    result = (
        X_train_enc.values.astype(np.float32),
        y_train,
        X_val_enc.values.astype(np.float32),
        y_val,
        X_test_enc.values.astype(np.float32),
        y_test,
        X_train_enc.columns,
    )
    return (*result, multi_mappings)



def load_and_prepare_datasets(
    data_dir: str = "datasets",
    prefix: str = "churn",
    target_col: str = "Churn",
    train_suffix: str = "_train.csv",
    val_suffix: str = "_val.csv",
    test_suffix: str = "_test.csv",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Dict[int, Any]],
]:
    train_df, val_df, test_df = load_split_datasets(data_dir=data_dir, prefix=prefix, 
                                                    train_suffix=train_suffix, val_suffix=val_suffix, test_suffix=test_suffix)

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, mappings = preprocess_data(
        train_df,
        val_df=val_df,
        test_df=test_df,
        target_col=target_col
    )
    train_df_prepared = pd.DataFrame(X_train, columns=feature_names)
    train_df_prepared[target_col] = y_train

    val_df_prepared = pd.DataFrame(X_val, columns=feature_names)
    val_df_prepared[target_col] = y_val

    test_df_prepared = pd.DataFrame(X_test, columns=feature_names)
    test_df_prepared[target_col] = y_test
    result = (
        train_df_prepared,
        val_df_prepared,
        test_df_prepared,
    )
    return (*result, mappings)
