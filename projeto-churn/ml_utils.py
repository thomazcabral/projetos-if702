import os
import math
import inspect
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

import matplotlib.pyplot as plt


EXPERIMENT_EPOCHS = 10000
RANDOM_STATE_SAMPLE = 42
RANDOM_STATE_MODEL = 42

DEFAULT_EPOCHS = 100
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_EARLY_STOPPING_DELTA = 0.001

SCORING_METRIC = "f1"


def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, pd.DataFrame) or isinstance(values, pd.Series):
        return values.values
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _encode_binary_labels(y: Any) -> np.ndarray:
    y_arr = _to_numpy(y)
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


def _logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    return torch.softmax(logits, dim=1)[:, 1]


def _predict_scores_torch(
    model: nn.Module,
    X: Any,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    X_np = _to_numpy(X).astype(np.float32)
    dataset = TensorDataset(torch.tensor(X_np, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            batch_scores = _logits_to_scores(logits)
            scores.append(batch_scores.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


def _predict_scores_proba(model: Any, X: Any) -> np.ndarray:
    proba = model.predict_proba(X)
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]
    return proba[:, 1]


def predict_scores(
    model: Any,
    X: Any,
    device: Optional[torch.device] = None,
    batch_size: int = 1024,
) -> np.ndarray:
    if isinstance(model, nn.Module):
        device = device or get_default_device()
        model.to(device)
        return _predict_scores_torch(model, X, device, batch_size)
    if hasattr(model, "predict_proba"):
        return _predict_scores_proba(model, X)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    raise ValueError("Model does not support prediction.")


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Optional[Any] = None,
) -> Dict[str, Any]:
    y_true_arr = _encode_binary_labels(y_true)
    y_pred_arr = _encode_binary_labels(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision_score(y_true_arr, y_pred_arr, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, zero_division=0),
    }

    if y_score is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true_arr, _to_numpy(y_score))
        except ValueError:
            metrics["auroc"] = None

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    return metrics


def evaluate_model(
    model: Any,
    X_train: Any,
    X_test: Optional[Any] = None,
    y_train: Optional[Any] = None,
    y_test: Optional[Any] = None,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 1024,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    if X_test is None:
        if y_train is None:
            raise ValueError("y_train is required when X_test is not provided.")
        scores = predict_scores(model, X_train, device=device, batch_size=batch_size)
        preds = (scores >= threshold).astype(int)
        metrics = compute_metrics(y_train, preds, scores)
        return metrics, preds, scores

    if y_train is None or y_test is None:
        raise ValueError("y_train and y_test are required when X_test is provided.")

    train_scores = predict_scores(model, X_train, device=device, batch_size=batch_size)
    train_preds = (train_scores >= threshold).astype(int)
    train_metrics = compute_metrics(y_train, train_preds, train_scores)

    test_scores = predict_scores(model, X_test, device=device, batch_size=batch_size)
    test_preds = (test_scores >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_preds, test_scores)

    if model_name:
        train_metrics["model"] = model_name
        test_metrics["model"] = model_name

    return train_metrics, test_metrics, test_preds


def ks_test(
    y_true: Any,
    y_score: Any,
    return_best: bool = False,
    plot: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Dict[str, Any]:
    if not return_best and not plot:
        raise ValueError("Select return_best and/or plot.")

    y_true_arr = _encode_binary_labels(y_true)
    y_score_arr = _to_numpy(y_score).astype(float)

    order = np.argsort(-y_score_arr)
    scores_sorted = y_score_arr[order]
    labels_sorted = y_true_arr[order]

    positives = labels_sorted == 1
    negatives = labels_sorted == 0

    if positives.sum() == 0 or negatives.sum() == 0:
        raise ValueError("Both classes are required to compute KS.")

    cum_pos = np.cumsum(positives) / positives.sum()
    cum_neg = np.cumsum(negatives) / negatives.sum()
    ks_values = np.abs(cum_pos - cum_neg)

    best_idx = int(np.argmax(ks_values))
    ks_stat = float(ks_values[best_idx])
    best_threshold = float(scores_sorted[best_idx])
    best_percentage = float((best_idx + 1) / len(scores_sorted) * 100.0)

    result = {"ks_stat": ks_stat}

    if return_best:
        result.update(
            {
                "best_threshold": best_threshold,
                "best_percentage": best_percentage,
            }
        )

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        population_pct = np.arange(1, len(scores_sorted) + 1) / len(scores_sorted) * 100.0
        ax.plot(population_pct, cum_pos, label="Cumulative positive")
        ax.plot(population_pct, cum_neg, label="Cumulative negative")
        ax.axvline(best_percentage, color="red", linestyle="--", label="Best KS")
        ax.set_xlabel("Population percentage")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title(f"KS statistic = {ks_stat:.4f}")
        ax.legend()
        result.update({"fig": fig, "ax": ax})

    return result


def get_loss_fn(loss_name: str = "cross_entropy", **kwargs: Any) -> nn.Module:
    if isinstance(loss_name, nn.Module):
        return loss_name
    name = str(loss_name).strip().lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    if name == "bce_with_logits":
        return nn.BCEWithLogitsLoss(**kwargs)
    raise ValueError(f"Unsupported loss: {loss_name}")


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
    RobustScaler,
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


def prepare_dataloaders(
    X_train: Any,
    y_train: Any,
    X_val: Optional[Any] = None,
    y_val: Optional[Any] = None,
    n_batches: int = 1,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], int]:
    X_train_np = _to_numpy(X_train).astype(np.float32)
    y_train_np = _encode_binary_labels(y_train).astype(np.int64)

    if batch_size is None:
        n_batches = max(1, int(n_batches))
        batch_size = max(1, int(math.ceil(len(X_train_np) / n_batches)))

    train_dataset = TensorDataset(
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_np = _to_numpy(X_val).astype(np.float32)
        y_val_np = _encode_binary_labels(y_val).astype(np.int64)
        val_batch_size = max(1, int(math.ceil(len(X_val_np) / n_batches)))
        val_dataset = TensorDataset(
            torch.tensor(X_val_np, dtype=torch.float32),
            torch.tensor(y_val_np, dtype=torch.long),
        )
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, batch_size


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    return running_loss / len(dataloader.dataset)


def _evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    model.eval()
    running_loss = 0.0
    scores = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            running_loss += loss.item() * batch_X.size(0)

            batch_scores = _logits_to_scores(outputs)
            scores.append(batch_scores.detach().cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss, np.concatenate(scores, axis=0)


def train_model(
    model: nn.Module,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    n_batches: int = 1,
    batch_size: Optional[int] = None,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    min_delta: float = DEFAULT_EARLY_STOPPING_DELTA,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    device = device or get_default_device()
    loss_fn = loss_fn or get_loss_fn()

    model.to(device)
    train_loader, val_loader, batch_size = prepare_dataloaders(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        n_batches=n_batches,
        batch_size=batch_size,
    )

    history = {"train_loss": [], "val_loss": [], "val_ks": []}
    best_metric = -np.inf
    best_state = None
    patience_counter = 0

    for _ in range(int(epochs)):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        history["train_loss"].append(train_loss)

        if val_loader is None:
            continue

        val_loss, val_scores = _evaluate_epoch(model, val_loader, loss_fn, device)
        history["val_loss"].append(val_loss)

        ks_result = ks_test(y_val, val_scores, return_best=True, plot=False)
        val_ks = ks_result["ks_stat"]
        history["val_ks"].append(val_ks)

        if val_ks > best_metric + min_delta:
            best_metric = val_ks
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_scores = predict_scores(model, X_val, device=device, batch_size=batch_size)
    final_preds = (final_scores >= threshold).astype(int)
    final_metrics = compute_metrics(y_val, final_preds, final_scores)

    return {
        "model": model,
        "history": history,
        "metrics": final_metrics,
        "preds": final_preds,
        "scores": final_scores,
        "best_metric": best_metric,
        "batch_size": batch_size,
    }


def infer(
    model: Any,
    X: Any,
    device: Optional[torch.device] = None,
    batch_size: int = 1024,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = predict_scores(model, X, device=device, batch_size=batch_size)
    preds = (scores >= threshold).astype(int)
    return preds, scores


def build_hyperparameter_space(
    model_class: Any,
    overrides: Optional[Dict[str, Iterable[Any]]] = None,
    include_params: Optional[Iterable[str]] = None,
    exclude_params: Optional[Iterable[str]] = None,
    extra_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    overrides = overrides or {}
    include_set = set(include_params) if include_params else None
    exclude_set = set(exclude_params) if exclude_params else set()

    space: Dict[str, Dict[str, Any]] = {}
    signature = inspect.signature(model_class.__init__)

    for name, param in signature.parameters.items():
        if name == "self":
            continue
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue

        param_type = None
        values = []

        if param.default is not inspect.Parameter.empty:
            param_type = type(param.default)
            values = [param.default]
        elif param.annotation is not inspect.Parameter.empty:
            param_type = param.annotation

        if name in overrides:
            values = list(overrides[name])

        space[name] = {"type": param_type, "values": values}

    if extra_params:
        for name, spec in extra_params.items():
            space[name] = {
                "type": spec.get("type"),
                "values": list(spec.get("values", [])),
            }

    if "n_batches" not in space:
        space["n_batches"] = {
            "type": int,
            "values": list(overrides.get("n_batches", [1])),
        }

    return space


def print_hyperparameter_space(space: Dict[str, Dict[str, Any]]) -> None:
    print("Hyperparameter space:")
    for name, spec in space.items():
        param_type = spec.get("type")
        values = spec.get("values")
        type_name = param_type.__name__ if param_type else "unknown"
        print(f"- {name}: type={type_name}, values={values}")


def _suggest_from_space(trial: Any, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params = {}
    for name, spec in space.items():
        values = list(spec.get("values", []))
        if not values:
            raise ValueError(f"Hyperparameter '{name}' has no candidate values.")
        params[name] = trial.suggest_categorical(name, values)
    return params


def _split_model_and_train_params(model_class: Any, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    signature = inspect.signature(model_class.__init__)
    init_params = set(signature.parameters.keys())
    init_params.discard("self")

    model_params = {}
    train_params = {}
    for name, value in params.items():
        if name in init_params:
            model_params[name] = value
        else:
            train_params[name] = value

    return model_params, train_params


def _is_torch_model_class(model_class: Any) -> bool:
    try:
        return issubclass(model_class, nn.Module)
    except TypeError:
        return False


def _is_xgboost_model_class(model_class: Any) -> bool:
    module_name = getattr(model_class, "__module__", "")
    return "xgboost" in module_name


def _stratified_sample(
    X: Any,
    y: Any,
    frac: Optional[float],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_np = _to_numpy(X)
    y_np = _encode_binary_labels(y)
    if frac is None or frac >= 1.0:
        return X_np, y_np

    X_sample, _, y_sample, _ = train_test_split(
        X_np,
        y_np,
        train_size=frac,
        stratify=y_np,
        random_state=random_state,
    )
    return X_sample, y_sample


def optuna_objective(
    trial: Any,
    model_class: Any,
    space: Dict[str, Dict[str, Any]],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    device: Optional[torch.device] = None,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    min_delta: float = DEFAULT_EARLY_STOPPING_DELTA,
    threshold: float = 0.5,
    sample_frac: Optional[float] = None,
    random_state: int = RANDOM_STATE_SAMPLE,
    loss_fn: Optional[nn.Module] = None,
    optimizer_class: Any = torch.optim.Adam,
) -> float:
    params = _suggest_from_space(trial, space)
    model_params, train_params = _split_model_and_train_params(model_class, params)

    n_batches = train_params.pop("n_batches", 1)
    batch_size = train_params.pop("batch_size", None)
    lr = train_params.pop("lr", train_params.pop("learning_rate", 1e-3))
    weight_decay = train_params.pop("weight_decay", 0.0)

    X_train_s, y_train_s = _stratified_sample(X_train, y_train, sample_frac, random_state)
    X_val_s, y_val_s = _stratified_sample(X_val, y_val, sample_frac, random_state)

    if _is_torch_model_class(model_class):
        model = model_class(**model_params)
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_result = train_model(
            model,
            X_train_s,
            y_train_s,
            X_val_s,
            y_val_s,
            optimizer,
            loss_fn=loss_fn,
            device=device,
            n_batches=n_batches,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
            threshold=threshold,
        )
        scores = train_result["scores"]
        ks_result = ks_test(y_val_s, scores, return_best=True, plot=False)
        return ks_result["ks_stat"]

    if _is_xgboost_model_class(model_class):
        model = model_class(**model_params)
        try:
            model.fit(
                X_train_s,
                y_train_s,
                eval_set=[(X_val_s, y_val_s)],
                early_stopping_rounds=patience,
                verbose=False,
            )
        except TypeError:
            model.fit(X_train_s, y_train_s)

        scores = predict_scores(model, X_val_s)
        ks_result = ks_test(y_val_s, scores, return_best=True, plot=False)
        return ks_result["ks_stat"]

    raise ValueError("Unsupported model class for Optuna objective.")


def run_optuna_search(
    model_class: Any,
    space: Dict[str, Dict[str, Any]],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    n_trials: int,
    direction: str = "maximize",
    **objective_kwargs: Any,
) -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for hyperparameter search.") from exc

    def _objective(trial: Any) -> float:
        return optuna_objective(
            trial,
            model_class,
            space,
            X_train,
            y_train,
            X_val,
            y_val,
            **objective_kwargs,
        )

    study = optuna.create_study(direction=direction)
    study.optimize(_objective, n_trials=int(n_trials))
    return study


def plot_metrics_history(
    history: Dict[str, Iterable[float]],
    metrics: Optional[Iterable[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if metrics is None:
        metrics = list(history.keys())

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for key in metrics:
        values = history.get(key, [])
        ax.plot(range(1, len(values) + 1), values, label=key)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training history")
    ax.legend()
    return fig, ax


def plot_roc_curve(
    y_true: Any,
    y_score: Any,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    y_true_arr = _encode_binary_labels(y_true)
    y_score_arr = _to_numpy(y_score)

    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    auc_value = roc_auc_score(y_true_arr, y_score_arr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.plot(fpr, tpr, label=f"AUROC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return fig, ax


def plot_score_boxplot(
    y_true: Any,
    y_score: Any,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    y_true_arr = _encode_binary_labels(y_true)
    y_score_arr = _to_numpy(y_score)

    scores_0 = y_score_arr[y_true_arr == 0]
    scores_1 = y_score_arr[y_true_arr == 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.boxplot([scores_0, scores_1], labels=["0", "1"], showmeans=True)
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted score")
    ax.set_title("Score distribution by class")
    return fig, ax


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Optional[Iterable[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    y_true_arr = _encode_binary_labels(y_true)
    y_pred_arr = _encode_binary_labels(y_pred)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    tick_labels = labels if labels is not None else ["0", "1"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    return fig, ax
