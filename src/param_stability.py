"""Parameter stability utilities."""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def save_fold_params(params_per_fold: List[Dict[str, float]], path: str) -> None:
    """Save best parameters from each fold to CSV."""
    df = pd.DataFrame(params_per_fold)
    logger.debug("Saving fold parameters to %s", path)
    df.to_csv(path, index=False)


def analyze_param_stability(
    params_per_fold: List[Dict[str, float]], threshold: float = 0.2
) -> pd.DataFrame:
    """Analyze parameter stability across folds.

    Parameters
    ----------
    params_per_fold : list of dict
        Best parameters from each fold.
    threshold : float, optional
        Relative standard deviation threshold to flag instability.

    Returns
    -------
    pd.DataFrame
        Table containing mean, std, relative std and ``unstable`` flag.
    """
    if not params_per_fold:
        raise ValueError("params_per_fold is empty")

    df = pd.DataFrame(params_per_fold)
    stats = df.agg(["mean", "std"]).transpose()
    stats.rename(columns={"mean": "mean", "std": "std"}, inplace=True)
    stats["rel_std"] = (stats["std"] / stats["mean"].abs()).fillna(0.0)
    stats["unstable"] = stats["rel_std"] > threshold
    stats.index.name = "param"
    return stats.reset_index()
