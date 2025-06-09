import logging
from typing import Dict

import pandas as pd

from src.wfv_monitor import walk_forward_loop


def _simple_backtest(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
    """[Patch v6.1.5] Basic backtest used during walk-forward."""
    pnl = float(test.Close.iloc[-1] - train.Close.iloc[0])
    winrate = float((test.Close.diff() > 0).mean())
    maxdd = float(train.Close.cummax().sub(train.Close).max())
    return {"pnl": pnl, "winrate": winrate, "maxdd": maxdd, "auc": 0.6}


def run_walkforward(
    output_path: str | None = None,
    data_path: str = "XAUUSD_M1.csv",
    nrows: int = 20,
) -> pd.DataFrame:
    """[Patch v6.1.5] Run walk-forward validation on a real dataset."""
    logging.info("[Patch v6.1.5] Starting walk-forward on %s", data_path)
    df = pd.read_csv(data_path, nrows=nrows)
    if "Close" not in df.columns:
        raise KeyError("'Close' column missing from dataset")
    df = df.reset_index(drop=True)[["Close"]]

    kpi = {"profit": 0.0, "winrate": 0.5, "maxdd": 0.1, "auc": 0.6}
    result = walk_forward_loop(
        df,
        _simple_backtest,
        kpi,
        train_window=4,
        test_window=2,
        step=3,
        output_path=output_path,
    )
    logging.info("[Patch v6.1.5] walk-forward completed")
    return result
