# -*- coding: utf-8 -*-
"""[Patch v1.1.1] Training utilities for hyperparameter sweep."""
import os
import numpy as np
import pandas as pd
from joblib import dump
from nicegold.config import logger, USE_GPU_ACCELERATION
from nicegold.utils.model_utils import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from nicegold.utils import convert_thai_datetime
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - lightgbm optional
    LGBMClassifier = None

def save_model(model, output_dir: str, model_name: str) -> None:
    """[Patch v5.3.2] Save model or create QA log if model is None."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}.joblib")
    if model is None:
        logger.warning(
            f"[QA] No model was trained for {model_name}. Creating empty model QA file."
        )
        qa_path = os.path.join(output_dir, f"{model_name}_qa.log")
        with open(qa_path, "w", encoding="utf-8") as f:
            f.write("[QA] No model trained. Output not generated.\n")
    else:
        dump(model, path)
        logger.info(f"[QA] Model saved: {path}")

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - fallback if catboost missing
    CatBoostClassifier = None


# [Patch v5.3.4] Add seed argument for deterministic behavior
# [Patch v1.1.0] Real training function using CatBoost (or logistic regression fallback)
def real_train_func(
    output_dir: str,
    learning_rate: float = 0.01,
    depth: int = 6,
    iterations: int = 100,
    l2_leaf_reg: int | float | None = None,
    seed: int = 42,
    trade_log_path: str | None = None,
    m1_path: str | None = None,
) -> dict:
    """Train a simple model and return model path, used features and metrics."""
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(seed)  # [Patch v5.3.4] Ensure deterministic training

    if trade_log_path and m1_path and os.path.exists(trade_log_path) and os.path.exists(m1_path):
        # [Patch v5.4.3] Allow using real trade data when paths are provided
        trade_df = pd.read_csv(trade_log_path)
        m1_df = pd.read_csv(m1_path)
        feature_cols = m1_df.select_dtypes(include=[np.number]).columns.tolist()
        if not feature_cols:
            raise ValueError("No numeric columns found in m1 data")
        min_len = min(len(trade_df), len(m1_df))
        X = m1_df.loc[:min_len-1, feature_cols].to_numpy()
        if 'profit' in trade_df.columns:
            y_raw = trade_df.loc[:min_len-1, 'profit']
        elif 'pnl_usd_net' in trade_df.columns:
            y_raw = trade_df.loc[:min_len-1, 'pnl_usd_net']
        else:
            num_cols = trade_df.select_dtypes(include=[np.number]).columns
            if num_cols.empty:
                raise ValueError("No numeric target column found in trade log")
            y_raw = trade_df.loc[:min_len-1, num_cols[0]]
        y = (y_raw > 0).astype(int).to_numpy()
        feature_names = feature_cols
    else:
        raise FileNotFoundError("ต้องส่งไฟล์ trade log จริงมา")

    df_X = pd.DataFrame(X, columns=feature_names)
    # [Patch v5.4.5] Use stratified split when possible to avoid ROC AUC warnings
    unique, counts = np.unique(y, return_counts=True)
    stratify_arg = y if (len(unique) > 1 and counts.min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        df_X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=stratify_arg,
    )

    if CatBoostClassifier:
        cat_params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "verbose": False,
            "task_type": "GPU" if USE_GPU_ACCELERATION else "CPU",
            "random_seed": seed,
        }
        if l2_leaf_reg is not None:
            cat_params["l2_leaf_reg"] = l2_leaf_reg  # pragma: no cover - catboost parameter
        model = CatBoostClassifier(**cat_params)  # pragma: no cover - optional catboost path
        model.fit(X_train, y_train)  # pragma: no cover - optional catboost path
        y_prob = model.predict_proba(X_test)[:, 1]  # pragma: no cover - optional catboost path
        y_pred = (y_prob > 0.5).astype(int)  # pragma: no cover - optional catboost path
    else:  # pragma: no cover - logistic fallback
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(X_train, y_train)  # pragma: no cover - sklearn deterministic
        y_prob = model.predict_proba(X_test)[:, 1]  # pragma: no cover - sklearn deterministic
        y_pred = model.predict(X_test)  # pragma: no cover - sklearn deterministic

    acc = accuracy_score(y_test, y_pred)
    # [Patch v5.4.5] Avoid UndefinedMetricWarning when only one class present
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = float('nan')

    param_suffix = f"_l2{l2_leaf_reg}" if l2_leaf_reg is not None else ""
    model_filename = f"model_lr{learning_rate}_depth{depth}{param_suffix}"
    model_path = os.path.join(output_dir, f"{model_filename}.joblib")
    save_model(model, output_dir, model_filename)

    return {
        "model_path": {"model": model_path},
        "features": feature_names,
        "metrics": {"accuracy": acc, "auc": auc},
    }


def optuna_sweep(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 20,
    output_path: str = "output_default/meta_classifier_optuna.pkl",
) -> dict:
    """ปรับ Hyperparameter ด้วย Optuna และบันทึกโมเดล"""
    from nicegold.config import optuna as _optuna

    if _optuna is None:
        logger.error("optuna not available")
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
        if USE_GPU_ACCELERATION:
            params["n_jobs"] = -1
        model = RandomForestClassifier(**params)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        model.fit(X_train, y_train)
        acc, auc = evaluate_model(model, X_test, y_test)
        return auc if not np.isnan(auc) else acc

    study = _optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X, y)
    save_model(best_model, os.path.dirname(output_path), os.path.splitext(os.path.basename(output_path))[0])
    return best_params


def _time_series_cv_auc(model_cls, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> float:
    """Return average AUC using TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            aucs.append(0.5)
            continue
        model = model_cls(random_state=random_state)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def train_lightgbm_mtf(m1_path: str, m15_path: str, output_dir: str, auc_threshold: float = 0.7) -> dict | None:
    """Train LightGBM model using merged M1+M15 features."""
    if LGBMClassifier is None:
        logger.error("lightgbm not available")
        return None

    if not os.path.exists(m1_path) or not os.path.exists(m15_path):
        logger.error("M1 or M15 data not found")
        return None

    df_m1 = pd.read_csv(m1_path)
    df_m15 = pd.read_csv(m15_path)
    df_m1 = convert_thai_datetime(df_m1, "timestamp")
    df_m15 = convert_thai_datetime(df_m15, "timestamp")
    df_m1.sort_values("timestamp", inplace=True)
    df_m15.sort_values("timestamp", inplace=True)
    df_m15["EMA_Fast"] = df_m15["Close"].ewm(span=50, adjust=False).mean()
    df_m15["EMA_Slow"] = df_m15["Close"].ewm(span=200, adjust=False).mean()
    df_m15["Trend_Up"] = (df_m15["EMA_Fast"] > df_m15["EMA_Slow"]).astype(int)
    merged = pd.merge_asof(
        df_m1,
        df_m15[["timestamp", "Close", "Trend_Up"]].rename(columns={"Close": "M15_Close"}),
        on="timestamp",
        direction="backward",
    ).dropna()
    merged["target"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)
    merged.dropna(inplace=True)
    features = ["Open", "High", "Low", "Close", "Volume", "M15_Close", "Trend_Up"]
    X = merged[features]
    y = merged["target"]

    avg_auc = _time_series_cv_auc(LGBMClassifier, X, y)
    logger.info(f"[QA] LightGBM CV AUC={avg_auc:.4f}")
    if avg_auc < auc_threshold:
        logger.error("AUC below threshold %.2f", auc_threshold)
        return None

    final_model = LGBMClassifier(random_state=42)
    final_model.fit(X, y)
    save_model(final_model, output_dir, "lightgbm_mtf")
    path = os.path.join(output_dir, "lightgbm_mtf.joblib")
    return {"model_path": {"model": path}, "features": features, "metrics": {"auc": avg_auc}}
