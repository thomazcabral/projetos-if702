import math
import inspect
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    roc_curve,
    confusion_matrix,
)
from scikitplot.helpers import binary_ks_curve

import matplotlib.pyplot as plt


TRAIN_EPOCHS = 10000
RANDOM_STATE_SAMPLE = 42
RANDOM_STATE_MODEL = 42

SEARCH_EPOCHS = 100
DEFAULT_EARLY_STOPPING_PATIENCE = 10
DEFAULT_EARLY_STOPPING_DELTA = 0.001

AVAILABLE_METRICS_SEARCH = ["accuracy", "precision", "recall", "f1", "val_loss", "auroc", "ks"]
AVAILABLE_LOSSES = ["cross_entropy", "bce_with_logits", "mse"]
DEFAULT_SCORING_METRIC = "ks"



### Helper functions gerais

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



#### Predição e avaliação de modelos

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


def _evaluate_loss(
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


def predict_scores(
    model: Any,
    X: Any,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
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

def ks_test(
    y_true: Any,
    y_score: Any
) -> Dict[str, Any]:
    """Calcula a estatística KS, retornando também o melhor limiar e a porcentagem correspondente"""

    y_true_arr = _encode_binary_labels(y_true)
    y_score_arr = _to_numpy(y_score).astype(float)

    (
        thresholds,
        pct1,
        pct2,
        ks_statistic,
        max_distance_at,
        _,
    ) = binary_ks_curve(y_true_arr, y_score_arr)

    idxs = np.where(thresholds == max_distance_at)[0]
    idx = int(idxs[0]) if len(idxs) else int(np.argmax(pct1 - pct2))

    n0 = int(np.sum(y_true_arr == 0))
    n1 = int(np.sum(y_true_arr == 1))
    total = n0 + n1
    if total > 0:
        best_percentage = float(((pct1[idx] * n0 + pct2[idx] * n1) / total) * 100.0)
    else:
        best_percentage = 0.0

    probas = np.column_stack([1.0 - y_score_arr, y_score_arr])
    result = {"ks_stat": float(ks_statistic), "probas": probas}


    result.update(
        {
            "best_threshold": float(max_distance_at),
            "best_percentage": best_percentage,
        }
    )

    return result

def compute_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Optional[Any] = None,
) -> Dict[str, Any]:
    """Retorna um dicionário com métricas de avaliação para classificação binária, incluindo acurácia, precisão, recall, F1, AUROC e KS."""
    y_true_arr = _encode_binary_labels(y_true)
    y_pred_arr = _encode_binary_labels(y_pred)
    auroc = None
    ks_stat = None
    if y_score is not None:
        y_score_arr = _to_numpy(y_score)
        auroc = roc_auc_score(y_true_arr, y_score_arr)
        ks_stat = ks_test(y_true_arr, y_score_arr)["ks_stat"]
        
    metrics = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision_score(y_true_arr, y_pred_arr, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, zero_division=0),
        "auroc": auroc,
        "ks_stat": ks_stat,
    }

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
    X: Any,
    y: Optional[Any] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Retorna um dicionário com métricas de avaliação, as previsões binárias e os scores de probabilidade para um modelo de classificação binária"""
    if X is None or y is None:
        raise ValueError("X and y are required for evaluation.")

    scores = predict_scores(model, X, device=device, batch_size=batch_size)
    preds = (scores >= threshold).astype(int)
    metrics = compute_metrics(y, preds, scores)

    return metrics, preds, scores

def get_loss_fn(loss_name: str = "cross_entropy", **kwargs: Any) -> nn.Module:
    if isinstance(loss_name, nn.Module):
        return loss_name
    name = str(loss_name).strip().lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)
    if name == "bce_with_logits":
        return nn.BCEWithLogitsLoss(**kwargs)
    if name == "mse":
        return nn.MSELoss(**kwargs)
    raise ValueError(f"Unsupported loss: {loss_name}")



### Preparação de dataloaders para PyTorch

def prepare_dataloader(
    X: Any,
    y: Any,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    X_np = _to_numpy(X).astype(np.float32)
    y_np = _encode_binary_labels(y).astype(np.int64)

    dataset = TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.long),
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader





#### Treinamento e inferência de modelos PyTorch

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


def train_model(
    model: nn.Module,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    epochs: int = TRAIN_EPOCHS,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    min_delta: float = DEFAULT_EARLY_STOPPING_DELTA,
) -> Dict[str, Any]:
    """Recebe também os dados de validação para calcular a perda de validação ao final do treinamento e retornar junto com o modelo treinado. 
    O modelo retornado é o melhor encontrado durante o processo, considerando early stopping."""
    
    device = device or get_default_device()
    loss_fn = loss_fn or get_loss_fn()

    model.to(device)
    train_loader = prepare_dataloader(X_train, y_train, batch_size=batch_size,)
    val_loader = prepare_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    history = []
    best_loss = np.inf
    best_state = None
    patience_counter = 0
    epochs_loss = 0
    
    for _ in range(int(epochs)):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        history.append(train_loss)
        epochs_loss += train_loss
        avg_loss = epochs_loss / len(history)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    val_loss, val_scores = _evaluate_loss(model, val_loader, loss_fn, device)
    
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "best_train_loss": best_loss,
        "val_loss": val_loss,
        "val_scores": val_scores,
        "batch_size": batch_size,
    }


def infer_class(
    model: Any,
    X: Any,
    device: Optional[torch.device] = None,
    batch_size: int = 1024,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = predict_scores(model, X, device=device, batch_size=batch_size)
    preds = (scores >= threshold).astype(int)
    return preds, scores





##### Busca de hiperparâmetros com Optuna



#### Construção do espaço de hiperparametros

def _build_param_spec(values: Any) -> Dict[str, Any]:
    if isinstance(values, dict) and "type" in values:
        kind = str(values.get("type")).strip().lower()
        if kind == "categorical":
            choices = values.get("values")
            if choices is None:
                choices = values.get("choices")
            if choices is None:
                choices = values.get("list")
            if choices is None:
                raise ValueError("categorical requires a list of values.")
            return {"suggest_type": "categorical", "values": list(choices)}

        if kind in {"int", "float"}:
            args = values.get("values")
            if args is None:
                low = values.get("low")
                high = values.get("high")
                if low is None or high is None:
                    raise ValueError("int/float requires low and high.")
                args = [low, high, values.get("step"), values.get("log")]

            if not isinstance(args, (list, tuple)) or len(args) < 2:
                raise ValueError("int/float values must be [low, high, step?, log?].")

            low = args[0]
            high = args[1]
            step = args[2] if len(args) > 2 else None
            log = args[3] if len(args) > 3 else False
            spec = {"suggest_type": kind, "args": [low, high], "kwargs": {}}
            if step is not None:
                spec["kwargs"]["step"] = step
            if log:
                spec["kwargs"]["log"] = bool(log)
            return spec

        raise ValueError(f"Unsupported suggest type: {values.get('type')}")
    raise ValueError("Each hyperparameter must be a dict with a 'type' field.")


def build_hyperparameter_space(
    model_class: Any,
    overrides: Optional[Dict[str, Iterable[Any]]] = None,
    include_params: Optional[Iterable[str]] = None,
    exclude_params: Optional[Iterable[str]] = None,
    extra_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Modelos que usam pytorch e otimizadores customizáveis devem incluir, no espaço de busca, os parâmetros de otimização:
        - lr
        - weight_decay
        - optimizer
        - batch_size      
    Usando os mesmos nomes acima.
    Modelos compatíveis com sklearn podem ser otimizados usando parâmetros de modelo e métricas de avaliação padrão.
    """
    overrides = overrides or {}
    
    if isinstance(include_params, dict):
        overrides = {**overrides, **include_params}
        include_set = set(include_params.keys())
    else:
        include_set = set(include_params) if include_params else None
    exclude_set = set(exclude_params) if exclude_params else set()

    space: Dict[str, Dict[str, Any]] = {}

    if include_set is None and not overrides:
        raise ValueError("Provide overrides or include_params to build the space.")

    signature = inspect.signature(model_class.__init__)
    names = [name for name in (include_set or overrides.keys()) if name not in exclude_set]

    for name in names:
        if name in overrides:
            space[name] = _build_param_spec(overrides[name])
            continue

        param = signature.parameters.get(name)
        if param is None or param.default is inspect.Parameter.empty:
            raise ValueError(f"Missing override spec for '{name}'.")

        space[name] = _build_param_spec({"type": "categorical", "values": [param.default]})

    if extra_params:
        for name, spec in extra_params.items():
            space[name] = _build_param_spec(spec)

    return space


def print_hyperparameter_space(space: Dict[str, Dict[str, Any]]) -> None:
    print("Hyperparameter space:")
    for name, spec in space.items():
        kind = spec.get("suggest_type")
        if kind == "categorical":
            print(f"- {name}: type=categorical, values={spec.get('values')}")
        else:
            args = spec.get("args", [])
            kwargs = spec.get("kwargs", {})
            print(f"- {name}: type={kind}, args={args}, kwargs={kwargs}")



### Funções auxiliares para sugestão correta de parametros e avaliação de métricas durante a busca 

def _suggest_from_space(trial: Any, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params = {}
    for name, spec in space.items():
        if "suggest_type" in spec:
            kind = spec["suggest_type"]
            if kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec["values"])
                continue
            if kind == "int":
                params[name] = trial.suggest_int(name, *spec["args"], **spec.get("kwargs", {}))
                continue
            if kind == "float":
                params[name] = trial.suggest_float(name, *spec["args"], **spec.get("kwargs", {}))
                continue
            raise ValueError(f"Unsupported suggest type: {kind}")
        raise ValueError(f"Missing suggest_type for '{name}'.")
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


def _is_sklearn_compatible_model_class(model_class: Any) -> bool:
    """Modelos que usam fit e predict_proba, como XGBoost, LightGBM, CatBoost e muitos modelos do scikit-learn."""
    module_name = getattr(model_class, "__module__", "")
    return (
        module_name.startswith("sklearn.")
        or module_name.startswith("xgboost.")
        or module_name.startswith("lightgbm.")
        or module_name.startswith("catboost.")
    )


def _resolve_objective_score(
    y_true: Any,
    y_score: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    scoring_metric: Any,
    val_loss: Optional[float] = None,
) -> float:

    name = str(scoring_metric).strip().lower()
    if name in {"val_loss", "loss"}:
        if val_loss is None:
            raise ValueError("val_loss is required for loss-based scoring.")
        return float(val_loss)
    if name in {"ks", "ks_test", "ks_statistic"}:
        if y_score is None:
            raise ValueError("y_score is required for KS.")
        return float(ks_test(y_true, y_score)["ks_stat"])
    if name in {"auroc", "roc_auc", "roc-auc"}:
        if y_score is None:
            raise ValueError("y_score is required for AUROC.")
        return float(roc_auc_score(_encode_binary_labels(y_true), _to_numpy(y_score)))
    if y_pred is None:
        raise ValueError("y_pred is required for classification metrics.")

    if name == "accuracy":
        return float(accuracy_score(_encode_binary_labels(y_true), _encode_binary_labels(y_pred)))
    if name == "precision":
        return float(precision_score(_encode_binary_labels(y_true), _encode_binary_labels(y_pred), zero_division=0))
    if name == "recall":
        return float(recall_score(_encode_binary_labels(y_true), _encode_binary_labels(y_pred), zero_division=0))
    if name == "f1":
        return float(f1_score(_encode_binary_labels(y_true), _encode_binary_labels(y_pred), zero_division=0))

    raise ValueError(f"Unsupported scoring metric: {scoring_metric}")


def _normalize_loss_name(loss_fn: Any) -> Optional[str]:
    if loss_fn is None:
        return None
    if isinstance(loss_fn, str):
        return loss_fn.strip().lower()
    if isinstance(loss_fn, nn.Module):
        name = loss_fn.__class__.__name__.lower()
    else:
        name = str(loss_fn).strip().lower()

    if name in {"crossentropyloss", "bcewithlogitsloss", "bceloss"}:
        return "cross_entropy"
    if name in {"mseloss", "mse"}:
        return "mse"
    return name


def _compute_sklearn_loss(y_true: Any, y_score: np.ndarray, loss_name: Optional[str]) -> Optional[float]:
    if loss_name is None:
        return None
    y_true_arr = _encode_binary_labels(y_true)
    if loss_name in {"cross_entropy", "bce_with_logits"}:
        return float(log_loss(y_true_arr, y_score))
    if loss_name == "mse":
        return float(mean_squared_error(y_true_arr, y_score))
    raise ValueError(f"Unsupported loss for sklearn models: {loss_name}")




### Funções do Optuna para executar a busca propriamente dita

def optuna_objective(
    trial: Any,
    model_class: Any,
    space: Dict[str, Dict[str, Any]],
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    scoring_metric: Any = DEFAULT_SCORING_METRIC,
    device: Optional[torch.device] = None,
    epochs: int = SEARCH_EPOCHS,
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    min_delta: float = DEFAULT_EARLY_STOPPING_DELTA,
    threshold: float = 0.5,
    loss_fn: Optional[nn.Module] = None,
    optimizer_class: Any = torch.optim.Adam,
) -> float:
    params = _suggest_from_space(trial, space)
    model_params, train_params = _split_model_and_train_params(model_class, params)

    batch_size = train_params.pop("batch_size", None)
    lr = train_params.pop("lr", train_params.pop("learning_rate", 1e-3))
    weight_decay = train_params.pop("weight_decay", 0.0)
    optimizer_choice = train_params.pop("optimizer", train_params.pop("optimizer_class", None))
    if optimizer_choice is not None:
        if isinstance(optimizer_choice, str):
            if hasattr(torch.optim, optimizer_choice):
                optimizer_class = getattr(torch.optim, optimizer_choice)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_choice}")
        else:
            optimizer_class = optimizer_choice

    is_torch = _is_torch_model_class(model_class)
    is_sklearn = _is_sklearn_compatible_model_class(model_class)
    if not (is_torch or is_sklearn):
        raise ValueError("Unsupported model class for Optuna objective.")

    model = model_class(**model_params)
    scores = None
    preds = None
    val_loss = None

    if is_torch:
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_result = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            optimizer,
            loss_fn=loss_fn,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
        )
        metrics, preds, scores = evaluate_model(
            model,
            X_val,
            y_val,
            device=device,
            batch_size=batch_size or 32,
            threshold=threshold,
        )
        val_loss = train_result.get("val_loss")

    if is_sklearn:
        model.fit(X_train, y_train)

        scores = predict_scores(model, X_val)
        preds = (scores >= threshold).astype(int)
        loss_name = _normalize_loss_name(loss_fn)
        val_loss = _compute_sklearn_loss(y_val, scores, loss_name)

    return _resolve_objective_score(y_val, scores, preds, scoring_metric, val_loss=val_loss,)


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





#### Plotagem
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
