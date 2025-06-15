# [Patch v6.9.6] Walk-forward result aggregator

import json
import os
import pandas as pd


def aggregate_wfv_results(result_dir: str) -> pd.DataFrame:
    """Combine out-of-sample results across all folds.

    Parameters
    ----------
    result_dir : str
        Path to directory containing ``fold_*`` subdirectories.

    Returns
    -------
    pd.DataFrame
        Combined trade log from all folds sorted by ``entry_time`` if present.
    """
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Directory not found: {result_dir}")

    fold_dirs = sorted(d for d in os.listdir(result_dir) if d.startswith("fold_"))
    if not fold_dirs:
        raise FileNotFoundError("No fold directories found")

    trade_logs = []
    for fd in fold_dirs:
        log_path = os.path.join(result_dir, fd, "oos_trade_log.csv")
        if os.path.exists(log_path):
            from src.utils.data_utils import safe_read_csv

            df = safe_read_csv(log_path)
            trade_logs.append(df)

    combined = (
        pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()
    )
    if "entry_time" in combined.columns:
        combined["entry_time"] = pd.to_datetime(combined["entry_time"], errors="coerce")
        combined.sort_values("entry_time", inplace=True)

    metrics = {
        "Total Net Profit": float(
            combined.get("pnl_usd_net", pd.Series(dtype=float)).sum()
        ),
        "Win Rate": float(
            (combined.get("pnl_usd_net", pd.Series(dtype=float)) > 0).mean() * 100.0
        ),
    }
    if not combined.empty and "pnl_usd_net" in combined.columns:
        equity = combined["pnl_usd_net"].cumsum()
        drawdown = equity.cummax() - equity
        metrics["Max Drawdown"] = float(drawdown.max())
    else:
        metrics["Max Drawdown"] = 0.0

    agg_log = os.path.join(result_dir, "full_oos_trade_log.csv")
    combined.to_csv(agg_log, index=False)
    with open(
        os.path.join(result_dir, "aggregated_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    return combined
