"""Utility functions for analyzing trade logs."""
# [Patch v5.5.16] Enhanced regex patterns and added export/plot helpers

from __future__ import annotations

import pandas as pd
import re
import logging
from datetime import datetime
from pathlib import Path



# [Patch] Regex patterns kept as constants for easier maintenance
ORDER_OPEN_PATTERN = re.compile(
    r"Open New Order.*?at (?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})"
)
ORDER_CLOSE_PATTERN = re.compile(
    r"Order Closing: Time=(?P<close>[^,]+), Final Reason=(?P<reason>[^,]+), ExitPrice=(?P<exit>[\d.]+), EntryTime=(?P<entry>[^,]+)"
)
PNL_PATTERN = re.compile(r"PnL\(Net USD\)=(?P<pnl>-?[\d.]+)")
ALERT_PATTERN = re.compile(r"^(?P<level>WARNING|ERROR|CRITICAL):[^:]*:(?P<msg>.*)$")


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
    path = Path(log_path)
    if path.suffix not in {".txt", ".log"}:
        logging.error("Invalid log file extension: %s", path.suffix)
        raise ValueError(f"Invalid log file extension: {path.suffix}")
    if not path.exists():
        logging.error("Log file not found: %s", log_path)
        raise FileNotFoundError(f"Log file not found: {log_path}")

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        line_buffer = []
        for chunk in iter(lambda: f.readlines(100_000), []):
            line_buffer.extend(chunk)

    i = 0
    while i < len(line_buffer):
        line = line_buffer[i]
        m_close = ORDER_CLOSE_PATTERN.search(line)
        if m_close:
            entry_time = datetime.fromisoformat(m_close.group("entry").strip())
            close_time = datetime.fromisoformat(m_close.group("close").strip())
            reason = m_close.group("reason").strip()
            pnl = None
            if i + 1 < len(line_buffer):
                m_pnl = PNL_PATTERN.search(line_buffer[i + 1])
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


def calculate_reason_summary(df: pd.DataFrame) -> pd.Series:
    """Return frequency count of close reasons."""
    if df.empty or "Reason" not in df:
        return pd.Series(dtype=int)
    return df["Reason"].value_counts()


def calculate_duration_stats(df: pd.DataFrame) -> dict[str, float]:
    """Compute statistics about trade duration in minutes."""
    if df.empty:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    durations = (df["CloseTime"] - df["EntryTime"]).dt.total_seconds() / 60.0
    return {
        "mean": durations.mean(),
        "median": durations.median(),
        "max": durations.max(),
    }


def calculate_drawdown_stats(df: pd.DataFrame) -> dict[str, float]:
    """Compute total PnL and maximum drawdown."""
    if df.empty or "PnL" not in df:
        return {"total_pnl": 0.0, "max_drawdown": 0.0}
    cumulative = df["PnL"].fillna(0).cumsum()
    equity = pd.concat([pd.Series([0.0]), cumulative], ignore_index=True)
    running_max = equity.cummax()
    drawdown = equity - running_max
    return {
        "total_pnl": df["PnL"].fillna(0).sum(),
        "max_drawdown": drawdown.min(),
    }


def calculate_expectancy(df: pd.DataFrame, pnl_col: str = "PnL") -> float:
    """Return expectancy from a series of trade PnL values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame ที่มีคอลัมน์กำไรขาดทุน
    pnl_col : str, optional
        ชื่อคอลัมน์ PnL ภายใน DataFrame

    Returns
    -------
    float
        ค่าคาดหวัง (Expectancy) ตามสูตร Win% * AvgWin - Loss% * AvgLoss
    """

    if df.empty or pnl_col not in df:
        return 0.0

    pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
    if pnl.empty:
        return 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_pct = len(wins) / len(pnl)
    loss_pct = len(losses) / len(pnl)
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0
    return float(win_pct * avg_win - loss_pct * avg_loss)


def parse_alerts(log_path: str) -> pd.DataFrame:
    """[Patch] Extract warning/error/critical messages from a log file."""
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = ALERT_PATTERN.match(line.strip())
            if m:
                entries.append({"level": m.group("level"), "message": m.group("msg").strip()})
    return pd.DataFrame(entries)


def calculate_alert_summary(log_path: str) -> pd.Series:
    """Return counts of WARNING/ERROR/CRITICAL messages."""
    df = parse_alerts(log_path)
    if df.empty:
        return pd.Series(dtype=int)
    return df["level"].value_counts()


def compile_log_summary(df: pd.DataFrame, log_path: str | None = None) -> dict[str, object]:
    """Return aggregate statistics for a parsed trade log and alert counts."""
    summary = {
        "hourly": calculate_hourly_summary(df),
        "reasons": calculate_reason_summary(df),
        "duration": calculate_duration_stats(df),
        "pnl": calculate_drawdown_stats(df),
    }
    if log_path:
        summary["alerts"] = calculate_alert_summary(log_path)
    return summary


def export_summary_to_csv(df: pd.DataFrame, output_path: str, compress: bool = True) -> None:
    """Export a DataFrame summary to CSV with optional gzip compression."""
    compression = "gzip" if compress else None
    df.to_csv(output_path, index=False, compression=compression)


def plot_summary(df: pd.DataFrame):
    """Return a matplotlib Figure of the hourly trade summary."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    df.plot(kind="bar", ax=ax)
    ax.set_xlabel("hour")
    ax.set_ylabel("value")
    return fig


def summarize_block_reasons(blocked_logs: list[dict]) -> pd.Series:
    """[Patch v5.7.3] Return counts of block reasons from blocked order log."""
    if not blocked_logs:
        return pd.Series(dtype=int)
    reasons = [b.get("reason", "UNKNOWN") for b in blocked_logs if isinstance(b, dict)]
    return pd.Series(reasons).value_counts()


# [Patch v6.1.6] Equity curve and expectancy analysis utilities
def calculate_equity_curve(df: pd.DataFrame, pnl_col: str = "PnL") -> pd.Series:
    """Return cumulative equity from trade PnL."""
    if df.empty or pnl_col not in df:
        return pd.Series(dtype=float)
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0)
    return pnl.cumsum()


def calculate_expectancy_by_period(
    df: pd.DataFrame, period: str = "H", pnl_col: str = "PnL"
) -> pd.Series:
    """Return expectancy grouped by time period (e.g., hourly)."""
    if df.empty or pnl_col not in df or "EntryTime" not in df:
        return pd.Series(dtype=float)
    df = df.dropna(subset=["EntryTime"]).copy()
    df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce")
    df = df.dropna(subset=[pnl_col])

    def _exp(x: pd.Series) -> float:
        wins = x[x > 0]
        losses = x[x <= 0]
        win_rate = (wins.count() / len(x)) if len(x) else 0.0
        avg_win = wins.mean() if not wins.empty else 0.0
        avg_loss = abs(losses.mean()) if not losses.empty else 0.0
        return float(win_rate * avg_win - (1 - win_rate) * avg_loss)

    grouped = df.groupby(df["EntryTime"].dt.to_period(period))[pnl_col]
    return grouped.apply(_exp)


def plot_equity_curve(curve: pd.Series):
    """Return a matplotlib Figure of the equity curve."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    curve.plot(ax=ax)
    ax.set_xlabel("trade")
    ax.set_ylabel("equity")
    return fig


# [Patch v6.1.7] Full trade log summarization helper
def summarize_trade_log(log_path: str) -> dict[str, object]:
    """Parse log and return summary with equity curve."""
    df = parse_trade_logs(log_path)
    summary = compile_log_summary(df, log_path)
    summary["equity_curve"] = calculate_equity_curve(df)
    summary["expectancy_H"] = calculate_expectancy_by_period(df)
    return summary

