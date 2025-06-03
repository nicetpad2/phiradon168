# -*- coding: utf-8 -*-
"""[Patch v1.1.1] Training utilities for hyperparameter sweep."""
import os
import numpy as np
import pandas as pd
from joblib import dump
from src.config import logger
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

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
    l2_leaf_reg: int | float | None = None,
    seed: int = 42,
) -> dict:
    """Train a simple model and return model path, used features and metrics."""
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(seed)  # [Patch v5.3.4] Ensure deterministic training

    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        random_state=seed,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    df_X = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        df_X,
        y,
        test_size=0.25,
        random_state=seed,
    )

    if CatBoostClassifier:
        cat_params = {
            "iterations": 100,
            "learning_rate": learning_rate,
            "depth": depth,
            "verbose": False,
        }
        if l2_leaf_reg is not None:
            cat_params["l2_leaf_reg"] = l2_leaf_reg
        model = CatBoostClassifier(**cat_params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
    else:
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    param_suffix = f"_l2{l2_leaf_reg}" if l2_leaf_reg is not None else ""
    model_filename = f"model_lr{learning_rate}_depth{depth}{param_suffix}"
    model_path = os.path.join(output_dir, f"{model_filename}.joblib")
    save_model(model, output_dir, model_filename)

    return {
        "model_path": {"model": model_path},
        "features": feature_names,
        "metrics": {"accuracy": acc, "auc": auc},
    }
