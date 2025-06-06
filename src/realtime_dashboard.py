import os
import time
import pandas as pd
from typing import Tuple
from src.dashboard import create_dashboard

try:  # pragma: no cover - optional dependency
    import streamlit as st
except Exception:  # pragma: no cover - fallback when streamlit missing
    st = None


def load_trade_log(csv_path: str) -> pd.DataFrame:
    """Load trade log CSV with required columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, parse_dates=["entry_time", "exit_time"], low_memory=False)
    if "pnl" not in df.columns:
        raise ValueError("trade log missing 'pnl' column")
    return df


def compute_equity_curve(trades: pd.DataFrame) -> pd.Series:
    """Return cumulative P&L as equity curve."""
    if trades.empty:
        return pd.Series(dtype=float)
    return trades["pnl"].cumsum()


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Calculate drawdown series from equity curve."""
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    return (equity - peak) / peak


def check_drawdown_alert(drawdown: pd.Series, threshold: float = 0.05) -> bool:
    """Return True if latest drawdown exceeds threshold."""
    if drawdown.empty:
        return False
    return bool(drawdown.iloc[-1] <= -abs(threshold))


def generate_dashboard(log_path: str, threshold: float = 0.05) -> Tuple[object, bool]:
    """Create dashboard figure and drawdown alert flag."""
    trades = load_trade_log(log_path)
    equity = compute_equity_curve(trades)
    dd = compute_drawdown(equity)
    fig = create_dashboard(equity, dd, trades["pnl"])
    return fig, check_drawdown_alert(dd, threshold)


def run_streamlit_dashboard(log_path: str, refresh_sec: int = 5, threshold: float = 0.05) -> None:
    """Start Streamlit app for real-time dashboard."""
    if st is None:
        raise ImportError("streamlit is required for dashboard")
    st.set_page_config(page_title="Real-Time Dashboard")
    placeholder = st.empty()
    while True:  # pragma: no cover - loop for UI
        fig, alert = generate_dashboard(log_path, threshold)
        placeholder.plotly_chart(fig, use_container_width=True)
        if alert:
            st.error(f"Drawdown exceeds {threshold*100:.1f}%!")
        time.sleep(refresh_sec)

__all__ = ["load_trade_log", "compute_equity_curve", "compute_drawdown", "check_drawdown_alert", "generate_dashboard", "run_streamlit_dashboard"]

