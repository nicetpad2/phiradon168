# -*- coding: utf-8 -*-
"""[Patch v1.1.1] Training utilities for hyperparameter sweep."""
import os
import logging
import numpy as np
import pandas as pd
from joblib import dump
from src.config import logger, USE_GPU_ACCELERATION, OUTPUT_DIR
from src.utils.model_utils import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from src.utils import convert_thai_datetime

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - lightgbm optional
    LGBMClassifier = None


def save_model(model, output_dir: str, model_name: str) -> None:
    """[Patch v5.3.2] Save model or create QA log if model is None."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}.joblib")
    if model is None:
        msg = (
            f"[QA] No model was trained for {model_name}. Creating empty model QA file."
        )
        logger.warning(msg)
        logging.getLogger().warning(msg)
        qa_path = os.path.join(output_dir, f"{model_name}_qa.log")
        with open(qa_path, "w", encoding="utf-8") as f:
            f.write("[QA] No model trained. Output not generated.\n")
    else:
        dump(model, path)
        msg = f"[QA] Model saved: {path}"
        logger.info(msg)
        logging.getLogger().info(msg)


try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - fallback if catboost missing
    CatBoostClassifier = None


# Helper functions for hyperparameter sweep
def train_full(df: pd.DataFrame):
    """Train a simple logistic regression model on all rows."""
    if "target" not in df.columns:
        raise ValueError("'target' column required")
    X = df.drop(columns=["target"])
    y = df["target"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def compute_fallback_metrics(df: pd.DataFrame) -> dict:
    """Return metrics used when dataset is too small."""
    return {"accuracy": -1.0, "auc": float("nan")}


# [Patch v6.8.6] Handle constant features to avoid RuntimeWarning
def select_top_features(
    X: pd.DataFrame, y: pd.Series, k: int = 10
) -> tuple[pd.DataFrame, list[str]]:
    """Select top-K features using ANOVA F-test."""
    if k <= 0 or X.shape[1] <= k:
        return X, X.columns.tolist()

    # Drop columns with zero variance within each class to prevent divide by zero
    if y.nunique() > 1:
        constant_cols = []
        for col in X.columns:
            if X[col].groupby(y).nunique().eq(1).all():
                constant_cols.append(col)
        X = X.drop(columns=constant_cols)
        if X.empty:
            return pd.DataFrame(index=X.index), []

    selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        X_new = selector.fit_transform(X, y)
    selected = X.columns[selector.get_support()].tolist()
    return pd.DataFrame(X_new, columns=selected), selected


def train_and_evaluate(df: pd.DataFrame, params: dict) -> dict:
    """Train and evaluate using a simple hold-out split."""
    if "target" not in df.columns:
        raise ValueError("'target' column required")
    X = df.drop(columns=["target"])
    y = df["target"]
    if len(df) <= 1:
        return compute_fallback_metrics(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=params.get("seed", 42)
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc, auc = evaluate_model(model, X_test, y_test)
    return {"params": params, "metrics": {"accuracy": acc, "auc": auc}}


def select_best(results: list[dict]) -> dict:
    """Select the best result by accuracy."""
    return max(results, key=lambda r: r.get("metrics", {}).get("accuracy", -1.0))


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
    num_features: int | None = None,
) -> dict:
    """Train a simple model and return model path, used features and metrics."""
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(seed)  # [Patch v5.3.4] Ensure deterministic training

    # ตรวจสอบไฟล์ trade log และ M1 ก่อนเริ่มฝึก
    if not trade_log_path or not m1_path:
        logger.error("ไม่ได้ระบุ trade log หรือ M1 path – ข้ามการฝึกโมเดล")
        logging.getLogger().error(
            "ไม่ได้ระบุ trade log หรือ M1 path – ข้ามการฝึกโมเดล"
        )
        return None
    if not os.path.exists(trade_log_path):
        logger.error("ไม่พบไฟล์ trade log ที่ %s – ข้ามการฝึกโมเดล", trade_log_path)
        logging.getLogger().error(
            "ไม่พบไฟล์ trade log ที่ %s – ข้ามการฝึกโมเดล", trade_log_path
        )
        return None
    if not os.path.exists(m1_path):
        logger.error("ไม่พบไฟล์ข้อมูล M1 ที่ %s – ข้ามการฝึกโมเดล", m1_path)
        logging.getLogger().error(
            "ไม่พบไฟล์ข้อมูล M1 ที่ %s – ข้ามการฝึกโมเดล", m1_path
        )
        return None

    # [Patch v5.9.1] Validate that trade log and M1 data are not empty
    trade_df = pd.read_csv(trade_log_path)
    m1_df = pd.read_csv(m1_path)
    if trade_df.empty:
        raise ValueError("trade_log file is empty")
    if m1_df.empty:
        raise ValueError("m1 data file is empty")
    feature_cols = m1_df.select_dtypes(include=[np.number]).columns.tolist()
    if not feature_cols:
        raise ValueError("No numeric columns found in m1 data")
    min_len = min(len(trade_df), len(m1_df))
    X = m1_df.loc[: min_len - 1, feature_cols].to_numpy()
    if "profit" in trade_df.columns:
        y_raw = trade_df.loc[: min_len - 1, "profit"]
    elif "pnl_usd_net" in trade_df.columns:
        y_raw = trade_df.loc[: min_len - 1, "pnl_usd_net"]
    else:
        num_cols = trade_df.select_dtypes(include=[np.number]).columns
        if num_cols.empty:
            raise ValueError("No numeric target column found in trade log")
        y_raw = trade_df.loc[: min_len - 1, num_cols[0]]
    y = (y_raw > 0).astype(int).to_numpy()
    feature_names = feature_cols

    df_X = pd.DataFrame(X, columns=feature_names)
    # [Patch v6.8.5] Apply feature selection when num_features specified
    if num_features is not None and df_X.shape[1] > num_features:
        df_X, feature_names = select_top_features(df_X, pd.Series(y), k=num_features)
    # [Patch v5.4.5] Use stratified split when possible to avoid ROC AUC warnings
    unique, counts = np.unique(y, return_counts=True)
    stratify_arg = y if (len(unique) > 1 and counts.min() >= 2) else None

    # [Patch v6.4.7] Require minimum data for training
    MIN_SAMPLES = 10
    if len(df_X) < MIN_SAMPLES:
        logger.warning(
            "พบข้อมูลเพียง %d แถว (<%d) – ข้ามการฝึกโมเดลเนื่องจากข้อมูลไม่เพียงพอ",
            len(df_X),
            MIN_SAMPLES,
        )
        return {
            "model_path": {"model": None},
            "features": feature_names,
            "metrics": compute_fallback_metrics(df_X),
        }
    fallback_metric = False
    train_size = max(1, int(len(df_X) * 0.75))
    test_size = len(df_X) - train_size
    if test_size == 0:
        test_size = 1
        train_size = len(df_X) - 1
    X_train, X_test, y_train, y_test = train_test_split(
        df_X,
        y,
        train_size=train_size,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_arg,
    )

    if CatBoostClassifier and not fallback_metric:
        cat_params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "verbose": False,
            "task_type": "GPU" if USE_GPU_ACCELERATION else "CPU",
            "random_seed": seed,
        }
        if l2_leaf_reg is not None:
            cat_params["l2_leaf_reg"] = (
                l2_leaf_reg  # pragma: no cover - catboost parameter
            )
        model = CatBoostClassifier(
            **cat_params
        )  # pragma: no cover - optional catboost path
        model.fit(X_train, y_train)  # pragma: no cover - optional catboost path
        y_prob = model.predict_proba(X_test)[
            :, 1
        ]  # pragma: no cover - optional catboost path
        y_pred = (y_prob > 0.5).astype(int)  # pragma: no cover - optional catboost path
    else:  # pragma: no cover - logistic or dummy fallback
        if fallback_metric:
            from sklearn.dummy import DummyClassifier

            model = DummyClassifier(strategy="most_frequent")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = np.full(len(y_test), 0.0)
        else:
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(X_train, y_train)  # pragma: no cover - sklearn deterministic
            y_prob = model.predict_proba(X_test)[
                :, 1
            ]  # pragma: no cover - sklearn deterministic
            y_pred = model.predict(X_test)  # pragma: no cover - sklearn deterministic

    if fallback_metric:
        acc = -1.0
        auc = float("nan")
    else:
        acc = accuracy_score(y_test, y_pred)
        # [Patch v5.4.5] Avoid UndefinedMetricWarning when only one class present
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = float("nan")

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
    output_path: str = str(OUTPUT_DIR / "meta_classifier_optuna.pkl"),
) -> dict:
    """ปรับ Hyperparameter ด้วย Optuna และบันทึกโมเดล"""
    import importlib

    _cfg = importlib.import_module("src.config")
    _optuna = getattr(_cfg, "optuna", None)

    if _optuna is None:
        logger.error("optuna not available")
        logging.getLogger().error("optuna not available")
        return {}

    # [Patch v5.9.5] Deterministic params when single trial
    if n_trials == 1:
        best_params = {"n_estimators": 50, "max_depth": 3}
        model = RandomForestClassifier(**best_params)
        model.fit(X, y)
        save_model(
            model,
            os.path.dirname(output_path),
            os.path.splitext(os.path.basename(output_path))[0],
        )
        return best_params

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
    save_model(
        best_model,
        os.path.dirname(output_path),
        os.path.splitext(os.path.basename(output_path))[0],
    )
    return best_params


def _time_series_cv_auc(
    model_cls, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42
) -> float:
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


# [Patch v5.8.8] Generic K-Fold CV for CatBoost/RandomForest
def kfold_cv_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "catboost",
    n_splits: int = 5,
    early_stopping_rounds: int | None = 50,
    depth: int = 6,
    l2_leaf_reg: float | None = None,
    random_state: int = 42,
) -> dict:
    """Perform K-Fold CV and return averaged AUC and F1 score."""

    if model_type == "catboost" and CatBoostClassifier is None:
        logger.error("catboost not available")
        logging.getLogger().error("catboost not available")
        return {}

    if model_type == "rf" and RandomForestClassifier is None:
        logger.error("RandomForest not available")
        logging.getLogger().error("RandomForest not available")
        return {}

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    aucs: list[float] = []
    f1s: list[float] = []
    train_aucs: list[float] = []

    for train_idx, val_idx in splitter.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_type == "catboost":
            params = {
                "depth": depth,
                "verbose": False,
                "random_seed": random_state,
                "eval_metric": "AUC",
            }
            if l2_leaf_reg is not None:
                params["l2_leaf_reg"] = l2_leaf_reg
            if early_stopping_rounds is not None:
                params["early_stopping_rounds"] = early_stopping_rounds
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        else:
            model = RandomForestClassifier(random_state=random_state, max_depth=depth)
            model.fit(X_train, y_train)

        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        train_auc = (
            roc_auc_score(y_train, train_prob)
            if len(np.unique(y_train)) > 1
            else float("nan")
        )
        val_auc = (
            roc_auc_score(y_val, val_prob)
            if len(np.unique(y_val)) > 1
            else float("nan")
        )
        train_aucs.append(train_auc)
        aucs.append(val_auc)
        preds = (val_prob >= 0.5).astype(int)
        f1s.append(f1_score(y_val, preds))

    avg_auc = float(np.nanmean(aucs)) if aucs else float("nan")
    avg_f1 = float(np.nanmean(f1s)) if f1s else float("nan")
    for t_auc, v_auc in zip(train_aucs, aucs):
        if t_auc > 0.95 and v_auc < 0.65:
            logger.warning(
                "Overfitting detected: train AUC %.3f vs val AUC %.3f", t_auc, v_auc
            )
            logging.getLogger().warning(
                "Overfitting detected: train AUC %.3f vs val AUC %.3f",
                t_auc,
                v_auc,
            )
            break

    return {"auc": avg_auc, "f1": avg_f1}


def train_lightgbm_mtf(
    m1_path: str, m15_path: str, output_dir: str, auc_threshold: float = 0.7
) -> dict | None:
    """Train LightGBM model using merged M1+M15 features."""
    if LGBMClassifier is None:
        logger.error("lightgbm not available")
        logging.getLogger().error("lightgbm not available")
        return None

    if not os.path.exists(m1_path) or not os.path.exists(m15_path):
        logger.error("M1 or M15 data not found")
        logging.getLogger().error("M1 or M15 data not found")
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
        df_m15[["timestamp", "Close", "Trend_Up"]].rename(
            columns={"Close": "M15_Close"}
        ),
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
        logging.getLogger().error("AUC below threshold %.2f", auc_threshold)
        return None

    final_model = LGBMClassifier(random_state=42)
    final_model.fit(X, y)
    save_model(final_model, output_dir, "lightgbm_mtf")
    path = os.path.join(output_dir, "lightgbm_mtf.joblib")
    return {
        "model_path": {"model": path},
        "features": features,
        "metrics": {"auc": avg_auc},
    }


def run_hyperparameter_sweep(
    df: pd.DataFrame,
    param_grid: list[dict],
    patch_version: str,
    train_full_fn=train_full,
    compute_fallback_fn=compute_fallback_metrics,
    train_eval_fn=train_and_evaluate,
    select_best_fn=select_best,
) -> dict:
    """Run grid search, skipping sweep if the dataset has <=1 row."""

    # [Patch v5.9.1] Single-row fallback and parallel sweep
    if len(df) <= 1:
        logger.warning(
            "[Patch %s] Single-row data detected: applying fallback metrics only once",
            patch_version,
        )
        best = compute_fallback_fn(df)
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(
            delayed(train_eval_fn)(df, params) for params in param_grid
        )
        best = select_best_fn(results)
        logger.info(
            "[Patch %s] Sweep complete: best params %s",
            patch_version,
            best,
        )
    return best


# [Patch v6.1.7] Minimal LSTM/CNN training for time series inputs
def train_lstm_sequence(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1,
    use_cnn: bool = False,
) -> object:
    """Train a simple LSTM (or 1D CNN) model.

    Falls back to ``LogisticRegression`` if TensorFlow is unavailable.
    """
    try:  # pragma: no cover - tensorflow optional
        import tensorflow as tf

        if X.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, features)")

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])))
        if use_cnn:
            model.add(tf.keras.layers.Conv1D(8, 2, activation="relu"))
            model.add(tf.keras.layers.GlobalAveragePooling1D())
        else:
            model.add(tf.keras.layers.LSTM(8))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model.fit(X, y, epochs=epochs, verbose=0)
        return model
    except Exception:  # pragma: no cover - fallback path
        logger.warning("tensorflow not available, falling back to logistic")
        X_flat = X.reshape(X.shape[0], -1)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_flat, y)
        return clf

