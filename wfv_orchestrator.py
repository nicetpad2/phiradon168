import logging
from typing import Iterable, Tuple

import os
import json
import pandas as pd
from src.csv_validator import validate_csv_data

try:
    from sklearn.model_selection import TimeSeriesSplit
except Exception as exc:  # pragma: no cover - optional dependency
    TimeSeriesSplit = None
    logging.error("scikit-learn unavailable: %s", exc)


# [Patch v6.5.9] Walk-forward orchestration helper
# Dynamic split count based on dataset length

def orchestrate_walk_forward(data: pd.DataFrame, n_splits: int = 5) -> Iterable[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield train/test pairs while restricting each fold to past data only."""
    if TimeSeriesSplit is None:
        raise ImportError("scikit-learn not installed")
    if not hasattr(data, "shape"):
        raise TypeError("data must be DataFrame-like")

    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    # Compute splits based on data length (at least 2 samples per fold)
    max_splits = min(n_splits, max(1, data.shape[0] // 2))
    kf = TimeSeriesSplit(n_splits=max_splits)
    for train_idx, test_idx in kf.split(data):
        end_label = data.index[test_idx[-1]]
        limited = data.loc[:end_label]
        yield limited.iloc[train_idx], limited.iloc[test_idx]

# [Patch v6.9.6] Minimal walk-forward runner with aggregation

def run_wfv_simple(data: pd.DataFrame, output_dir: str, n_splits: int = 5) -> None:
    """Run a minimal walk-forward loop and aggregate results."""
    validate_csv_data(data, required_cols=None)
    os.makedirs(output_dir, exist_ok=True)
    for i, (train_df, test_df) in enumerate(orchestrate_walk_forward(data, n_splits)):
        fold_dir = os.path.join(output_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        # build simple feature using only data up to this fold
        fold_end = test_df.index[-1]
        data_until_now = data.loc[:fold_end].copy()
        data_until_now["diff"] = data_until_now.Close.diff().fillna(0)
        train_feat = data_until_now.loc[train_df.index]
        test_feat = data_until_now.loc[test_df.index]

        # toy model uses mean diff as signal
        _model = train_feat["diff"].mean()
        pnl = float(test_feat.Close.iloc[-1] - train_feat.Close.iloc[0])
        pd.DataFrame({"entry_time": [int(fold_end)], "pnl_usd_net": [pnl]}).to_csv(
            os.path.join(fold_dir, "oos_trade_log.csv"), index=False
        )
        with open(os.path.join(fold_dir, "oos_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"total_net_profit": pnl}, f)
    from src.wfv_aggregator import aggregate_wfv_results
    aggregate_wfv_results(output_dir)
