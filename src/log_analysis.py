"""Utility functions for analyzing trade logs."""

from __future__ import annotations

import pandas as pd
import re
from datetime import datetime
from typing import Iterable


LOG_OPEN_RE = re.compile(r"Open New Order.*?at (?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})")
LOG_CLOSE_RE = re.compile(
    r"Order Closing: Time=(?P<close>[^,]+), Final Reason=(?P<reason>[^,]+), ExitPrice=(?P<exit>[\d.]+), EntryTime=(?P<entry>[^,]+)"
)
LOG_PNL_RE = re.compile(r"PnL\(Net USD\)=(?P<pnl>-?[\d.]+)")


def parse_trade_logs(log_path: str) -> pd.DataFrame:
    """Parse a log file and extract trade events.

    Parameters
    ----------
    log_path : str
        Path to the log file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns EntryTime, CloseTime, Reason, PnL.
    """
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        m_close = LOG_CLOSE_RE.search(line)
        if m_close:
            entry_time = datetime.fromisoformat(m_close.group("entry").strip())
            close_time = datetime.fromisoformat(m_close.group("close").strip())
            reason = m_close.group("reason").strip()
            pnl = None
            if i + 1 < len(lines):
                m_pnl = LOG_PNL_RE.search(lines[i + 1])
                if m_pnl:
                    pnl = float(m_pnl.group("pnl"))
                    i += 1
            entries.append(
                {
                    "EntryTime": entry_time,
                    "CloseTime": close_time,
                    "Reason": reason,
                    "PnL": pnl,
                }
            )
        i += 1
    df = pd.DataFrame(entries)
    if not df.empty:
        df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=True)
        df["CloseTime"] = pd.to_datetime(df["CloseTime"], utc=True)
    return df

def calculate_hourly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return win rate and average PnL per hour of entry."""
    if df.empty:
        return pd.DataFrame(columns=["count", "win_rate", "avg_pnl"])
    df = df.dropna(subset=["EntryTime", "PnL"])
    df["hour"] = df["EntryTime"].dt.hour
    grouped = df.groupby("hour")
    summary = pd.DataFrame()
    summary["count"] = grouped.size()
    summary["win_rate"] = grouped["PnL"].apply(lambda x: (x > 0).mean())
    summary["avg_pnl"] = grouped["PnL"].mean()
    return summary


def calculate_position_size(capital: float, risk_pct: float, stop_loss_pips: float, pip_value: float = 1.0) -> float:
    """Calculate lot size based on risk percentage and stop loss distance."""
    if capital <= 0 or risk_pct <= 0 or stop_loss_pips <= 0:
        raise ValueError("Input values must be positive")
    risk_amount = capital * (risk_pct / 100.0)
    position_units = risk_amount / (stop_loss_pips * pip_value)
    return position_units / 100000  # standard lot size

