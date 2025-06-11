import logging
from typing import Iterable, Tuple

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
