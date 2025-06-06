"""Signal classification utilities with basic feature engineering.

[Patch v5.9.0] Initial implementation for simple ML signal classifier.
"""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
import numpy as np

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

__all__ = [
    "create_label_from_backtest",
    "add_basic_features",
    "train_signal_classifier",
    "shap_feature_analysis",
    "tune_threshold_optuna",
    "train_meta_model",
]


def create_label_from_backtest(df: pd.DataFrame, target_point: float = 5) -> pd.DataFrame:
    """Create binary labels from future price movement."""
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    if not {"High", "Close"}.issubset(df.columns):
        raise KeyError("DataFrame missing required columns")

    df = df.copy()
    df["future_max"] = df["High"].shift(-5).rolling(5).max()
    df["label"] = ((df["future_max"] - df["Close"]) >= target_point).astype(int)
    return df.drop(columns="future_max")


def _rolling_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple technical indicators to DataFrame."""
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    required = {"Close", "High", "Low", "Volume"}
    if not required.issubset(df.columns):
        raise KeyError(f"Missing columns: {required - set(df.columns)}")

    df = df.copy()
    df["price_change"] = df["Close"].pct_change().fillna(0)
    df["atr"] = _rolling_atr(df["High"], df["Low"], df["Close"], window=14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["volume_change"] = df["Volume"].pct_change().fillna(0)
    df["ma_fast"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["ma_slow"] = df["Close"].rolling(window=30, min_periods=1).mean()
    df["ma_cross"] = (df["ma_fast"] > df["ma_slow"]).astype(int)
    return df


def train_signal_classifier(df: pd.DataFrame) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, np.ndarray]:
    """Train a RandomForest classifier on basic features."""
    feature_cols = [
        "price_change",
        "atr",
        "rsi",
        "volume_change",
        "ma_fast",
        "ma_slow",
        "ma_cross",
    ]
    if "label" not in df.columns:
        raise KeyError("DataFrame must contain 'label'")

    X = df[feature_cols].fillna(0)
    y = df["label"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    val_proba = clf.predict_proba(X_val)[:, 1]
    return clf, X_val, y_val, val_proba


def shap_feature_analysis(model: RandomForestClassifier, X_val: pd.DataFrame) -> np.ndarray | None:
    """Return SHAP values for validation set if shap library available."""
    if shap is None:
        logger.warning("shap library not available")
        return None
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)[1]
    return shap_values


def tune_threshold_optuna(y_true: pd.Series, y_proba: np.ndarray, n_trials: int = 50) -> float:
    """Tune probability threshold using Optuna to maximise F1 score."""
    try:
        import optuna  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("optuna not available; returning default threshold 0.5")
        return 0.5

    def objective(trial: "optuna.trial.Trial") -> float:
        thresh = trial.suggest_float("threshold", 0.3, 0.9)
        preds = (y_proba >= thresh).astype(int)
        return f1_score(y_true, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return float(study.best_params["threshold"])


def train_meta_model(base_signals_df: pd.DataFrame) -> LogisticRegression:
    """Train a simple logistic regression meta model."""
    required = {"macro_trend", "micro_trend", "ml_signal", "label"}
    if not required.issubset(base_signals_df.columns):
        missing = required - set(base_signals_df.columns)
        raise KeyError(f"Missing columns: {missing}")

    X = base_signals_df[["macro_trend", "micro_trend", "ml_signal"]]
    y = base_signals_df["label"]
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model
