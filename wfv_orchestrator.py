import logging
from typing import Iterable, Tuple

import os
import json
import pandas as pd

try:
    from sklearn.model_selection import TimeSeriesSplit
except Exception as exc:  # pragma: no cover - optional dependency
    TimeSeriesSplit = None
    logging.error("scikit-learn unavailable: %s", exc)


# [Patch v6.5.9] Walk-forward orchestration helper
# Dynamic split count based on dataset length

def orchestrate_walk_forward(data: pd.DataFrame, n_splits: int = 5) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield train/test pairs using TimeSeriesSplit."""
    if TimeSeriesSplit is None:
        raise ImportError("scikit-learn not installed")
    if not hasattr(data, "shape"):
        raise TypeError("data must be DataFrame-like")

    # Compute splits based on data length (at least 2 samples per fold)
    max_splits = min(n_splits, max(1, data.shape[0] // 2))
    kf = TimeSeriesSplit(n_splits=max_splits)
    for train_idx, test_idx in kf.split(data):
        yield data.iloc[train_idx], data.iloc[test_idx]

# [Patch v6.9.6] Minimal walk-forward runner with aggregation

def run_wfv_simple(data: pd.DataFrame, output_dir: str, n_splits: int = 5) -> None:
    """Run a minimal walk-forward loop and aggregate results."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (train_df, test_df) in enumerate(orchestrate_walk_forward(data, n_splits)):
        fold_dir = os.path.join(output_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        pnl = float(test_df.Close.iloc[-1] - train_df.Close.iloc[0])
        pd.DataFrame({"entry_time": [0], "pnl_usd_net": [pnl]}).to_csv(
            os.path.join(fold_dir, "oos_trade_log.csv"), index=False
        )
        with open(os.path.join(fold_dir, "oos_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"total_net_profit": pnl}, f)
    from src.wfv_aggregator import aggregate_wfv_results
    aggregate_wfv_results(output_dir)
