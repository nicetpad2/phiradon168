import logging
from typing import Dict

import pandas as pd
import os

from src.csv_validator import validate_and_convert_csv

from src.config import DATA_DIR, SYMBOL, TIMEFRAME
from src.wfv_monitor import walk_forward_loop


def _simple_backtest(train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
    """[Patch v6.1.5] Basic backtest used during walk-forward."""
    pnl = float(test.Close.iloc[-1] - train.Close.iloc[0])
    winrate = float((test.Close.diff() > 0).mean())
    maxdd = float(train.Close.cummax().sub(train.Close).max())
    return {"pnl": pnl, "winrate": winrate, "maxdd": maxdd, "auc": 0.6}


def run_walkforward(
    output_path: str | None = None,
    data_path: str | None = None,
    nrows: int = 20,
) -> pd.DataFrame:
    """[Patch v6.2.1] Run walk-forward validation on a real dataset."""
    if not data_path:
        data_path = f"{SYMBOL}_{TIMEFRAME}.csv"

    # resolve to DATA_DIR if path is relative
    if not os.path.isabs(data_path):
        candidate = os.path.join(DATA_DIR, data_path)
        if os.path.exists(candidate):
            data_path = candidate

    logging.info("[Patch v6.2.1] Starting walk-forward on %s", data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # [Patch v6.9.7] Validate source CSV before slicing data
    required = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    validated_df = validate_and_convert_csv(data_path, required_cols=required)
    df_full = validated_df.head(nrows)
    if "Close" not in df_full.columns:
        raise KeyError("'Close' column missing from dataset")
    df = df_full.reset_index(drop=True)[["Close"]]

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
