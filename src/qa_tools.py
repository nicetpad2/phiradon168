import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from src import strategy

logger = logging.getLogger(__name__)


def quick_qa_output(output_dir: str = "output_default", report_file: str = "qa_report.txt"):
    """Scan output files and report folds without trades or missing columns."""
    issues = []
    p = Path(output_dir)
    for f in p.glob("*.csv.gz"):
        try:
            df = pd.read_csv(f)
            missing_cols = [c for c in ["pnl", "entry_price"] if c not in df.columns]
            if df.empty:
                issues.append(f"{f.name}: No trades")
            elif missing_cols:
                issues.append(f"{f.name}: Missing columns {','.join(missing_cols)}")
        except Exception as e:
            issues.append(f"{f.name}: Error {e}")
    report_path = p / report_file
    with open(report_path, "w", encoding="utf-8") as fh:
        for line in issues:
            fh.write(line + "\n")
    logger.info("QA report written to %s", report_path)
    return issues


def run_noise_backtest(n: int = 1000, initial_price: float = 1800.0,
                       vol: float = 0.1, seed: int | None = None,
                       initial_capital: float = 10000.0,
                       **kwargs):
    """Generate random-walk price data and run backtest on the noise."""
    if seed is not None:
        np.random.seed(seed)
    price = initial_price + np.cumsum(np.random.randn(n) * vol)
    idx = pd.date_range(start="2020-01-01", periods=n, freq="T")
    df_noise = pd.DataFrame({
        "Open": price,
        "High": price + np.abs(np.random.randn(n) * vol),
        "Low": price - np.abs(np.random.randn(n) * vol),
        "Close": price,
        "ATR_14_Shifted": vol,
    }, index=idx)

    results = strategy.run_backtest_simulation_v34(
        df_noise, "NOISE", initial_capital, **kwargs
    )
    trade_log = results[1]
    final_equity = results[2]

    total_pnl = final_equity - initial_capital
    winrate = 0.0
    if isinstance(trade_log, pd.DataFrame) and not trade_log.empty:
        pnl_col = pd.to_numeric(
            trade_log.get("pnl_usd_net", trade_log.get("pnl", 0)),
            errors="coerce",
        ).fillna(0.0)
        winrate = float((pnl_col > 0).mean() * 100.0)

    return {
        "df": df_noise,
        "total_pnl": float(total_pnl),
        "winrate": winrate,
    }


__all__ = ["quick_qa_output", "run_noise_backtest"]
