import pandas as pd
import numpy as np
import logging
from catboost import CatBoostClassifier
from datetime import datetime
from typing import Dict, List

try:
    import numba
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba unavailable
    numba = None
    def njit(func):
        return func

# --- เตรียม Cache/Model Instances ---
_catboost_model_cache: Dict[str, CatBoostClassifier] = {}

# --- Numba-accelerated core OMS/backtest loop ---
@njit
def _run_oms_backtest_numba(prices: np.ndarray,
                            highs: np.ndarray,
                            lows: np.ndarray,
                            open_signals: np.ndarray,
                            close_signals: np.ndarray,
                            sl_prices: np.ndarray,
                            tp_prices: np.ndarray) -> np.int64:
    """Loop เปิด/ปิด Orders แบบเร่งด้วย Numba"""
    trades_executed = 0
    for i in range(prices.shape[0]):
        if open_signals[i] == 1:
            trades_executed += 1
        elif close_signals[i] == 1:
            trades_executed += 1
    return trades_executed


def run_all_folds_with_threshold(df_all: pd.DataFrame, folds: List[tuple]) -> Dict[int, int]:
    """เรียกใช้ backtest แบบง่ายด้วยฟังก์ชัน Numba"""
    results: Dict[int, int] = {}
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        df_backtest = df_all.iloc[test_idx].copy()
        prices = df_backtest["Close"].to_numpy()
        highs = df_backtest["High"].to_numpy()
        lows = df_backtest["Low"].to_numpy()
        open_signals = generate_open_signals(df_backtest)
        close_signals = generate_close_signals(df_backtest)
        sl_prices = precompute_sl_array(df_backtest)
        tp_prices = precompute_tp_array(df_backtest)
        trades_count = _run_oms_backtest_numba(
            prices, highs, lows, open_signals, close_signals, sl_prices, tp_prices
        )
        logging.info(f"Fold {fold_idx} completed. Trades executed (Numba): {trades_count}")
        results[fold_idx] = trades_count
    logging.info("All folds finished.")
    return results


def generate_open_signals(df: pd.DataFrame) -> np.ndarray:
    """สร้างสัญญาณเปิด order"""
    return (df["Close"] > df["Close"].shift(1)).fillna(0).astype(np.int8).to_numpy()


def generate_close_signals(df: pd.DataFrame) -> np.ndarray:
    """สร้างสัญญาณปิด order"""
    return (df["Close"] < df["Close"].shift(1)).fillna(0).astype(np.int8).to_numpy()


def precompute_sl_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Stop-Loss ล่วงหน้า"""
    return np.zeros(len(df), dtype=np.float64)


def precompute_tp_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Take-Profit ล่วงหน้า"""
    return np.zeros(len(df), dtype=np.float64)
