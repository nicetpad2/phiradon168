import logging
from typing import Dict

import pandas as pd

from src.wfv_monitor import walk_forward_validate


def _simple_backtest(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
    """[Patch] Minimal backtest for walk-forward example."""
    pnl = float(test.Close.iloc[-1] - train.Close.iloc[0])
    return {"pnl": pnl, "winrate": 0.6, "maxdd": 0.05, "auc": 0.7}


def run_walkforward() -> pd.DataFrame:
    """Run a sample walk-forward validation using :mod:`wfv_monitor`."""
    logging.info("[Patch] Starting sample walk-forward")
    df = pd.DataFrame({"Close": range(50)})
    kpi = {"profit": 0.0, "winrate": 0.5, "maxdd": 0.1, "auc": 0.6}
    result = walk_forward_validate(df, _simple_backtest, kpi, n_splits=5)
    logging.info("[Patch] walk-forward completed")
    return result
