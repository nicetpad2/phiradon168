from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Tuple, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score

from nicegold import config
logger = config.logger


# [Patch v5.8.0] Joint optimization of model and strategy parameters using Optuna

def joint_optuna_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    model_class: Callable[..., Any],
    model_space: Dict[str, Tuple[float, float, float]],
    strategy_space: Dict[str, Tuple[float, float, float]],
    n_splits: int = 3,
    n_trials: int = 20,
    direction: str = "maximize",
) -> Tuple[float, Dict[str, Any]]:
    """Run Optuna optimization jointly over model and strategy params.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    model_class : Callable
        Model class to instantiate per trial.
    model_space : dict
        Parameter space for the model. ``(low, high, step)``.
    strategy_space : dict
        Parameter space for strategy params. ``(low, high, step)``.
    n_splits : int, optional
        Number of CV splits. Defaults to 3.
    n_trials : int, optional
        Number of Optuna trials. Defaults to 20.
    direction : str, optional
        Optimization direction. Defaults to "maximize".

    Returns
    -------
    Tuple[float, Dict[str, Any]]
        Best score and corresponding parameters.
    """

    optuna_lib = getattr(config, "optuna", None)
    if optuna_lib is None:  # pragma: no cover - optuna may not be installed
        logger.error("optuna not available")
        return 0.0, {}

    cv = TimeSeriesSplit(n_splits=n_splits)

    def suggest_params(space: Dict[str, Tuple[float, float, float]], trial: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, (low, high, step) in space.items():
            if isinstance(low, int) and isinstance(high, int) and float(step).is_integer():
                params[name] = trial.suggest_int(name, int(low), int(high), step=int(step))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high), step=step)
        return params

    def objective(trial: Any) -> float:
        model_params = suggest_params(model_space, trial)
        strategy_params = suggest_params(strategy_space, trial)
        threshold = strategy_params.get("threshold", 0.5)

        aucs = []
        f1s = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= threshold).astype(int)
            if len(set(y_test)) > 1:
                aucs.append(roc_auc_score(y_test, proba))
            f1s.append(f1_score(y_test, preds))
        mean_auc = float(pd.Series(aucs).mean()) if aucs else 0.0
        mean_f1 = float(pd.Series(f1s).mean()) if f1s else 0.0
        return (mean_auc + mean_f1) / 2

    study = optuna_lib.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    return study.best_value, study.best_params
