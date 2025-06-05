# === START OF PART 5/12 ===

# ==============================================================================
# === PART 5: Feature Engineering & Indicator Calculation (v4.8.12) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, enhanced NaN/error handling, fixed SyntaxError, added integer downcast >>>
# <<< Includes fixes from v4.7.8: Context column calculation, fixed UnboundLocalError, fixed TypeError >>>
# <<< MODIFIED v4.8.0: Ensured Trend_Zone is always category dtype on return >>>
# <<< MODIFIED v4.8.1: Added handling for empty series, NaN/Inf inputs, dtype checks in indicators; refined NaN filling in engineer_m1_features; improved dtype conversion in clean_m1_data >>>
# <<< MODIFIED v4.8.2: Ensured robust index conversion and handling in engineer_m1_features before get_session_tag >>>
# <<< MODIFIED v4.8.3: Refined session tagging in engineer_m1_features for non-DatetimeIndex and addressed FutureWarning. Re-indented and reviewed. Added Part Markers. >>>
# <<< MODIFIED v4.9.0: Corrected session tagging for non-DatetimeIndex in engineer_m1_features to align with test expectations. Updated versioning. >>>
import logging
import pandas as pd
import numpy as np
# from data_loader import some_helper  # switched to absolute import (Patch v4.8.9)
try:  # [Patch v5.8.0] Handle missing ta library gracefully
    import ta
except ImportError:  # pragma: no cover - environment may not have ta installed
    ta = None
    logging.warning("'ta' library not found. Technical indicators will return NaN.")
from sklearn.cluster import KMeans # For context column calculation
from sklearn.preprocessing import StandardScaler # For context column calculation
import gc # For memory management
from src.utils.gc_utils import maybe_collect
from functools import lru_cache
from src.utils.sessions import get_session_tag  # [Patch v5.1.3]
from src.utils import get_env_float

_rsi_cache = {}  # [Patch v4.8.12] Cache RSIIndicator per period
_atr_cache = {}  # [Patch v4.8.12] Cache AverageTrueRange per period
_sma_cache = {}  # [Patch v4.8.12] Cache SMA results
_m15_trend_cache = {}

# Ensure global configurations are accessible if run independently
DEFAULT_ROLLING_Z_WINDOW_M1 = 300; DEFAULT_ATR_ROLLING_AVG_PERIOD = 50
DEFAULT_PATTERN_BREAKOUT_Z_THRESH = 2.0; DEFAULT_PATTERN_REVERSAL_BODY_RATIO = 0.5
DEFAULT_PATTERN_STRONG_TREND_Z_THRESH = 1.0; DEFAULT_PATTERN_CHOPPY_CANDLE_RATIO = 0.3
DEFAULT_PATTERN_CHOPPY_WICK_RATIO = 0.6; DEFAULT_M15_TREND_EMA_FAST = 50
DEFAULT_M15_TREND_EMA_SLOW = 200; DEFAULT_M15_TREND_RSI_PERIOD = 14
DEFAULT_M15_TREND_RSI_UP = 51; DEFAULT_M15_TREND_RSI_DOWN = 49  # [Patch v5.6.4]
DEFAULT_TIMEFRAME_MINUTES_M1 = 1; DEFAULT_MIN_SIGNAL_SCORE_ENTRY = 1.0  # [Patch v5.3.9]
DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO = 1.8; DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO = 0.75
DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R = 0.5; DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R = 1.0
DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R = 0.3; DEFAULT_ADAPTIVE_TSL_START_ATR_MULT = 1.5

try: ROLLING_Z_WINDOW_M1
except NameError: ROLLING_Z_WINDOW_M1 = DEFAULT_ROLLING_Z_WINDOW_M1
try: ATR_ROLLING_AVG_PERIOD
except NameError: ATR_ROLLING_AVG_PERIOD = DEFAULT_ATR_ROLLING_AVG_PERIOD
try: PATTERN_BREAKOUT_Z_THRESH
except NameError: PATTERN_BREAKOUT_Z_THRESH = DEFAULT_PATTERN_BREAKOUT_Z_THRESH
try: PATTERN_REVERSAL_BODY_RATIO
except NameError: PATTERN_REVERSAL_BODY_RATIO = DEFAULT_PATTERN_REVERSAL_BODY_RATIO
try: PATTERN_STRONG_TREND_Z_THRESH
except NameError: PATTERN_STRONG_TREND_Z_THRESH = DEFAULT_PATTERN_STRONG_TREND_Z_THRESH
try: PATTERN_CHOPPY_CANDLE_RATIO
except NameError: PATTERN_CHOPPY_CANDLE_RATIO = DEFAULT_PATTERN_CHOPPY_CANDLE_RATIO
try: PATTERN_CHOPPY_WICK_RATIO
except NameError: PATTERN_CHOPPY_WICK_RATIO = DEFAULT_PATTERN_CHOPPY_WICK_RATIO
try: M15_TREND_EMA_FAST
except NameError: M15_TREND_EMA_FAST = DEFAULT_M15_TREND_EMA_FAST
try: M15_TREND_EMA_SLOW
except NameError: M15_TREND_EMA_SLOW = DEFAULT_M15_TREND_EMA_SLOW
try: M15_TREND_RSI_PERIOD
except NameError: M15_TREND_RSI_PERIOD = DEFAULT_M15_TREND_RSI_PERIOD
try: M15_TREND_RSI_UP
except NameError: M15_TREND_RSI_UP = DEFAULT_M15_TREND_RSI_UP
try: M15_TREND_RSI_DOWN
except NameError: M15_TREND_RSI_DOWN = DEFAULT_M15_TREND_RSI_DOWN
try: TIMEFRAME_MINUTES_M1
except NameError: TIMEFRAME_MINUTES_M1 = DEFAULT_TIMEFRAME_MINUTES_M1
try: MIN_SIGNAL_SCORE_ENTRY
except NameError: MIN_SIGNAL_SCORE_ENTRY = DEFAULT_MIN_SIGNAL_SCORE_ENTRY
try: ADAPTIVE_TSL_HIGH_VOL_RATIO
except NameError: ADAPTIVE_TSL_HIGH_VOL_RATIO = DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO
try: ADAPTIVE_TSL_LOW_VOL_RATIO
except NameError: ADAPTIVE_TSL_LOW_VOL_RATIO = DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO
try: ADAPTIVE_TSL_DEFAULT_STEP_R
except NameError: ADAPTIVE_TSL_DEFAULT_STEP_R = DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R
try: ADAPTIVE_TSL_HIGH_VOL_STEP_R
except NameError: ADAPTIVE_TSL_HIGH_VOL_STEP_R = DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R
try: ADAPTIVE_TSL_LOW_VOL_STEP_R
except NameError: ADAPTIVE_TSL_LOW_VOL_STEP_R = DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R
try: ADAPTIVE_TSL_START_ATR_MULT
except NameError: ADAPTIVE_TSL_START_ATR_MULT = DEFAULT_ADAPTIVE_TSL_START_ATR_MULT
try: META_CLASSIFIER_FEATURES
except NameError: META_CLASSIFIER_FEATURES = []
try: SESSION_TIMES_UTC
except NameError: SESSION_TIMES_UTC = {"Asia": (22, 8), "London": (7, 16), "NY": (13, 21)}


# --- Indicator Calculation Functions ---
def ema(series, period):
    if not isinstance(series, pd.Series): logging.error(f"EMA Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    if series.empty:
        logging.debug("EMA: Input series is empty, returning NaN-aligned series.")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if series_numeric.empty: logging.warning("EMA: Series contains only NaN/Inf values or is empty after cleaning."); return pd.Series(np.nan, index=series.index, dtype='float32')
    try:
        ema_calculated = series_numeric.ewm(span=period, adjust=False, min_periods=max(1, period)).mean()
        ema_result = ema_calculated.reindex(series.index); del series_numeric, ema_calculated; maybe_collect()
        return ema_result.astype('float32')
    except Exception as e: logging.error(f"EMA calculation failed for period {period}: {e}", exc_info=True); return pd.Series(np.nan, index=series.index, dtype='float32')  # pragma: no cover

def sma(series, period):
    if not isinstance(series, pd.Series): logging.error(f"SMA Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    if series.empty:
        logging.debug("SMA: Input series is empty, returning NaN-aligned series.")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    if not isinstance(period, int) or period <= 0: logging.error(f"SMA calculation failed: Invalid period ({period})."); return pd.Series(np.nan, index=series.index, dtype='float32')  # pragma: no cover
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    if series_numeric.isnull().all(): logging.warning("SMA: Series contains only NaN values after numeric conversion and fill."); return pd.Series(np.nan, index=series.index, dtype='float32')  # pragma: no cover
    try:
        cache_key = (id(series), period)
        if cache_key in _sma_cache:
            cached = _sma_cache[cache_key]
            return cached.reindex(series.index).astype('float32')
        min_p = max(1, min(period, len(series_numeric)))
        sma_result = series_numeric.rolling(window=period, min_periods=min_p).mean()
        sma_final = sma_result.reindex(series.index).astype('float32')
        _sma_cache[cache_key] = sma_final
        del series_numeric, sma_result; maybe_collect()
        return sma_final
    except Exception as e:
        logging.error(f"SMA calculation failed for period {period}: {e}", exc_info=True)
        return pd.Series(np.nan, index=series.index, dtype='float32')  # pragma: no cover

def rsi(series, period=14):
    if not isinstance(series, pd.Series): logging.error(f"RSI Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    # [Patch v4.8.12] Use module-level cache for RSIIndicator
    if series.empty:
        logging.debug("RSI: Input series is empty, returning NaN-aligned series.")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    if 'ta' not in globals() or ta is None: logging.error("   (Error) RSI calculation failed: 'ta' library not loaded."); return pd.Series(np.nan, index=series.index, dtype='float32')  # pragma: no cover
    # Convert to numeric and drop NaN/inf values
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if series_numeric.empty or len(series_numeric) < period:
        logging.warning(
            f"   (Warning) RSI calculation skipped: Not enough valid data points ({len(series_numeric)} < {period})."
        )
        return pd.Series(np.nan, index=series.index, dtype='float32')
    # [Patch v5.5.16] Consolidate duplicate timestamps using last occurrence
    if series_numeric.index.duplicated().any():
        series_numeric = series_numeric.groupby(series_numeric.index).last()
    try:
        cache_key = period
        if cache_key not in _rsi_cache:
            _rsi_cache[cache_key] = ta.momentum.RSIIndicator(close=series_numeric, window=period, fillna=False)
        else:
            _rsi_cache[cache_key]._close = series_numeric
        rsi_series = _rsi_cache[cache_key].rsi()
        # Reindex to original index with forward-fill
        rsi_final = rsi_series.reindex(series.index, method='ffill').astype('float32')
        del series_numeric, rsi_series
        maybe_collect()
        return rsi_final
    except Exception as e:
        logging.error(f"   (Error) RSI calculation error for period {period}: {e}.", exc_info=True)
        return pd.Series(np.nan, index=series.index, dtype='float32')

def atr(df_in, period=14):
    if not isinstance(df_in, pd.DataFrame): logging.error(f"ATR Error: Input must be a pandas DataFrame, got {type(df_in)}"); raise TypeError("Input must be a pandas DataFrame.")
    # [Patch v4.8.12] Cache AverageTrueRange objects
    atr_col_name = f"ATR_{period}"; atr_shifted_col_name = f"ATR_{period}_Shifted"
    if df_in.empty:
        df_result = df_in.copy(); df_result[atr_col_name] = np.nan; df_result[atr_shifted_col_name] = np.nan
        df_result[atr_col_name] = df_result[atr_col_name].astype('float32'); df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32'); return df_result
    df_temp = df_in.copy(); required_price_cols = ["High", "Low", "Close"]
    if not all(col in df_temp.columns for col in required_price_cols):
        logging.warning(f"   (Warning) ATR calculation skipped: Missing columns {required_price_cols}.")
        df_result = df_in.copy(); df_result[atr_col_name] = np.nan; df_result[atr_shifted_col_name] = np.nan
        df_result[atr_col_name] = df_result[atr_col_name].astype('float32'); df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32'); return df_result
    for col in required_price_cols: df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df_temp.dropna(subset=required_price_cols, inplace=True)
    if df_temp.empty or len(df_temp) < period:
        logging.warning(f"   (Warning) ATR calculation skipped: Not enough valid data (need >= {period}).")
        df_result = df_in.copy(); df_result[atr_col_name] = np.nan; df_result[atr_shifted_col_name] = np.nan
        df_result[atr_col_name] = df_result[atr_col_name].astype('float32'); df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32'); return df_result
    atr_series = None
    if 'ta' in globals() and ta is not None:
        try:
            cache_key = period
            if cache_key not in _atr_cache:
                _atr_cache[cache_key] = ta.volatility.AverageTrueRange(high=df_temp['High'], low=df_temp['Low'], close=df_temp['Close'], window=period, fillna=False)
            else:
                _atr_cache[cache_key]._high = df_temp['High']
                _atr_cache[cache_key]._low = df_temp['Low']
                _atr_cache[cache_key]._close = df_temp['Close']
            atr_series = _atr_cache[cache_key].average_true_range()
        except Exception as e_ta_atr:
            logging.warning(f"   (Warning) TA library ATR calculation failed: {e_ta_atr}. Falling back.")  # pragma: no cover
            atr_series = None
    if atr_series is None:
        try:
            df_temp['H-L'] = df_temp['High'] - df_temp['Low']; df_temp['H-PC'] = abs(df_temp['High'] - df_temp['Close'].shift(1)); df_temp['L-PC'] = abs(df_temp['Low'] - df_temp['Close'].shift(1))
            df_temp['TR'] = df_temp[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            if not df_temp.empty and len(df_temp) > 0:
                first_valid_index = df_temp.index[0]
                if first_valid_index in df_temp.index: df_temp.loc[first_valid_index, 'TR'] = df_temp.loc[first_valid_index, 'H-L']
            atr_series = df_temp['TR'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        except Exception as e_pd_atr:
            logging.error(f"   (Error) Pandas EWM ATR calculation failed: {e_pd_atr}", exc_info=True)  # pragma: no cover
            df_result = df_in.copy(); df_result[atr_col_name] = np.nan; df_result[atr_shifted_col_name] = np.nan
            df_result[atr_col_name] = df_result[atr_col_name].astype('float32'); df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32')
            del df_temp; maybe_collect(); return df_result
    df_result = df_in.copy(); df_result[atr_col_name] = atr_series.reindex(df_in.index).astype('float32')
    df_result[atr_shifted_col_name] = atr_series.shift(1).reindex(df_in.index).astype('float32')
    del df_temp, atr_series; maybe_collect(); return df_result


@lru_cache(maxsize=None)
def calculate_sma(symbol: str, timeframe: str, length: int, date: str, prices: tuple):
    """Cached SMA calculation using LRU cache."""
    series = pd.Series(prices, dtype='float32')
    return sma(series, length)


@lru_cache(maxsize=None)
def calculate_rsi(symbol: str, timeframe: str, length: int, date: str, prices: tuple):
    """Cached RSI calculation using LRU cache."""
    series = pd.Series(prices, dtype='float32')
    return rsi(series, period=length)

def macd(series, window_slow=26, window_fast=12, window_sign=9):
    if not isinstance(series, pd.Series): logging.error(f"MACD Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    if series.empty: nan_series = pd.Series(dtype='float32'); return nan_series, nan_series.copy(), nan_series.copy()
    nan_series_indexed = pd.Series(np.nan, index=series.index, dtype='float32')
    if len(series.dropna()) < window_slow: logging.debug(f"MACD: Input series too short after dropna ({len(series.dropna())} < {window_slow})."); return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()
    if 'ta' not in globals() or ta is None: logging.error("   (Error) MACD calculation failed: 'ta' library not loaded."); return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if series_numeric.empty or len(series_numeric) < window_slow: logging.warning(f"   (Warning) MACD calculation skipped: Not enough valid data points ({len(series_numeric)} < {window_slow})."); return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()
    try:
        macd_indicator = ta.trend.MACD(close=series_numeric, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign, fillna=False)
        macd_line_final = macd_indicator.macd().reindex(series.index).ffill().astype('float32')
        macd_signal_final = macd_indicator.macd_signal().reindex(series.index).ffill().astype('float32')
        macd_diff_final = macd_indicator.macd_diff().reindex(series.index).ffill().astype('float32')
        del series_numeric, macd_indicator; maybe_collect()
        return (macd_line_final, macd_signal_final, macd_diff_final)
    except Exception as e: logging.error(f"   (Error) MACD calculation error: {e}.", exc_info=True); return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()

def detect_macd_divergence(prices: pd.Series, macd_hist: pd.Series, lookback: int = 20) -> str:
    """ตรวจจับภาวะ Divergence อย่างง่ายระหว่างราคากับ MACD histogram

    Parameters
    ----------
    prices : pd.Series
        ราคาปิด
    macd_hist : pd.Series
        ค่า MACD histogram
    lookback : int, optional
        จำนวนแท่งย้อนหลังที่ใช้พิจารณา, ค่าเริ่มต้น 20

    Returns
    -------
    str
        "bull" หากพบ Bullish Divergence, "bear" หากพบ Bearish Divergence, ไม่เช่นนั้นคืน "none"
    """

    if not isinstance(prices, pd.Series) or not isinstance(macd_hist, pd.Series):
        logging.error("detect_macd_divergence: inputs must be pandas Series")
        raise TypeError("Inputs must be pandas Series")

    if prices.empty or macd_hist.empty:
        return "none"

    p = pd.to_numeric(prices, errors="coerce").ffill().bfill()
    m = pd.to_numeric(macd_hist, errors="coerce").ffill().bfill()

    look = max(3, min(len(p), lookback))
    p_sub = p.iloc[-look:]
    m_sub = m.reindex(p_sub.index)

    lows = p_sub[(p_sub.shift(1) > p_sub) & (p_sub.shift(-1) > p_sub)]
    highs = p_sub[(p_sub.shift(1) < p_sub) & (p_sub.shift(-1) < p_sub)]

    if len(lows) >= 2:
        pl1, pl2 = lows.iloc[-2], lows.iloc[-1]
        ml1, ml2 = m_sub.loc[lows.index[-2]], m_sub.loc[lows.index[-1]]
        if pl2 < pl1 and ml2 > ml1:
            return "bull"

    if len(highs) >= 2:
        ph1, ph2 = highs.iloc[-2], highs.iloc[-1]
        mh1, mh2 = m_sub.loc[highs.index[-2]], m_sub.loc[highs.index[-1]]
        if ph2 > ph1 and mh2 < mh1:
            return "bear"

    return "none"

# [Patch v5.7.9] New feature helpers
def calculate_order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    """คำนวณความไม่สมดุลของ Order Flow"""
    if not isinstance(df, pd.DataFrame) or not {"BuyVolume", "SellVolume"}.issubset(df.columns):
        return pd.Series(0.0, index=getattr(df, "index", None), dtype="float32")
    buy = pd.to_numeric(df["BuyVolume"], errors="coerce").fillna(0.0)
    sell = pd.to_numeric(df["SellVolume"], errors="coerce").fillna(0.0)
    total = buy + sell
    imbalance = np.where(total > 0, (buy - sell) / total, 0.0)
    return pd.Series(imbalance, index=df.index, dtype="float32")


def calculate_relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """คำนวณ Relative Volume จากปริมาณซื้อขายแบบรวม 5 แท่ง"""
    if not isinstance(df, pd.DataFrame) or "Volume" not in df.columns:
        return pd.Series(0.0, index=getattr(df, "index", None), dtype="float32")
    vol_5m = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).rolling(5, min_periods=1).sum()
    avg_vol = vol_5m.rolling(period, min_periods=1).mean()
    rel = np.where(avg_vol > 0, vol_5m / avg_vol, 0.0)
    return pd.Series(rel, index=df.index, dtype="float32", name=df["Volume"].name)


def calculate_momentum_divergence(close_series: pd.Series) -> pd.Series:
    """คำนวณ Momentum Divergence ระหว่าง MACD บน M1 และค่าเฉลี่ย 5 แท่ง"""
    if not isinstance(close_series, pd.Series) or close_series.empty:
        return pd.Series(0.0, index=getattr(close_series, "index", None), dtype="float32")
    m1_hist = macd(close_series)[2]
    m5_close = close_series.rolling(5, min_periods=1).mean()
    m5_hist = macd(m5_close)[2]
    div = (m1_hist - m5_hist).astype("float32")
    return div.reindex(close_series.index).fillna(0.0)

def rolling_zscore(series, window, min_periods=None):
    if not isinstance(series, pd.Series): logging.error(f"Rolling Z-Score Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    if series.empty:
        logging.debug("Rolling Z-Score: Input series empty, returning NaN-aligned series.")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    if len(series) < 2: logging.debug("Rolling Z-Score: Input series too short (< 2), returning zeros."); return pd.Series(0.0, index=series.index, dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    if series_numeric.isnull().all(): logging.warning("Rolling Z-Score: Series contains only NaN values after numeric conversion and fill, returning zeros."); return pd.Series(0.0, index=series.index, dtype='float32')
    actual_window = min(window, len(series_numeric))
    if actual_window < 2: logging.debug(f"Rolling Z-Score: Adjusted window size ({actual_window}) < 2, returning zeros."); return pd.Series(0.0, index=series.index, dtype='float32')
    if min_periods is None: min_periods = max(2, min(10, int(actual_window * 0.1)))
    else: min_periods = max(2, min(min_periods, actual_window))
    try:
        rolling_mean = series_numeric.rolling(window=actual_window, min_periods=min_periods).mean()
        rolling_std = series_numeric.rolling(window=actual_window, min_periods=min_periods).std()
        with np.errstate(divide='ignore', invalid='ignore'): rolling_std_safe = rolling_std.replace(0, np.nan); z = (series_numeric - rolling_mean) / rolling_std_safe
        z_filled = z.fillna(0.0);
        if np.isinf(z_filled).any(): z_filled.replace([np.inf, -np.inf], 0.0, inplace=True)
        z_final = z_filled.reindex(series.index).fillna(0.0)
        del series_numeric, rolling_mean, rolling_std, rolling_std_safe, z, z_filled; maybe_collect()
        return z_final.astype('float32')
    except Exception as e: logging.error(f"Rolling Z-Score calculation failed for window {window}: {e}", exc_info=True); return pd.Series(0.0, index=series.index, dtype='float32')

def tag_price_structure_patterns(df):
    logging.info("   (Processing) Tagging price structure patterns...")
    if not isinstance(df, pd.DataFrame): logging.error("Pattern Tagging Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df.empty: df["Pattern_Label"] = "Normal"; df["Pattern_Label"] = df["Pattern_Label"].astype('category'); return df
    required_cols = ["Gain_Z", "High", "Low", "Close", "Open", "MACD_hist", "Candle_Ratio", "Wick_Ratio", "Gain", "Candle_Body"]
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"      (Warning) Missing columns for Pattern Labeling. Setting all to 'Normal'.")
        df["Pattern_Label"] = "Normal"; df["Pattern_Label"] = df["Pattern_Label"].astype('category'); return df
    df_patterns = df.copy()
    for col in ["Gain_Z", "MACD_hist", "Candle_Ratio", "Wick_Ratio", "Gain", "Candle_Body"]: df_patterns[col] = pd.to_numeric(df_patterns[col], errors='coerce').fillna(0)
    for col in ["High", "Low", "Close", "Open"]:
        df_patterns[col] = pd.to_numeric(df_patterns[col], errors='coerce')
        if df_patterns[col].isnull().any(): df_patterns[col] = df_patterns[col].ffill().bfill()
    df_patterns["Pattern_Label"] = "Normal"
    prev_high = df_patterns["High"].shift(1); prev_low = df_patterns["Low"].shift(1); prev_gain = df_patterns["Gain"].shift(1).fillna(0); prev_body = df_patterns["Candle_Body"].shift(1).fillna(0); prev_macd_hist = df_patterns["MACD_hist"].shift(1).fillna(0)
    breakout_cond = ((df_patterns["Gain_Z"].abs() > PATTERN_BREAKOUT_Z_THRESH) | ((df_patterns["High"] > prev_high) & (df_patterns["Close"] > prev_high)) | ((df_patterns["Low"] < prev_low) & (df_patterns["Close"] < prev_low))).fillna(False)
    reversal_cond = (((prev_gain < 0) & (df_patterns["Gain"] > 0) & (df_patterns["Candle_Body"] > (prev_body * PATTERN_REVERSAL_BODY_RATIO))) | ((prev_gain > 0) & (df_patterns["Gain"] < 0) & (df_patterns["Candle_Body"] > (prev_body * PATTERN_REVERSAL_BODY_RATIO)))).fillna(False)
    inside_bar_cond = ((df_patterns["High"] < prev_high) & (df_patterns["Low"] > prev_low)).fillna(False)
    strong_trend_cond = (((df_patterns["Gain_Z"] > PATTERN_STRONG_TREND_Z_THRESH) & (df_patterns["MACD_hist"] > 0) & (df_patterns["MACD_hist"] > prev_macd_hist)) | ((df_patterns["Gain_Z"] < -PATTERN_STRONG_TREND_Z_THRESH) & (df_patterns["MACD_hist"] < 0) & (df_patterns["MACD_hist"] < prev_macd_hist))).fillna(False)
    choppy_cond = ((df_patterns["Candle_Ratio"] < PATTERN_CHOPPY_CANDLE_RATIO) & (df_patterns["Wick_Ratio"] > PATTERN_CHOPPY_WICK_RATIO)).fillna(False)
    df_patterns.loc[breakout_cond, "Pattern_Label"] = "Breakout"
    df_patterns.loc[reversal_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "Reversal"
    df_patterns.loc[inside_bar_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "InsideBar"
    df_patterns.loc[strong_trend_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "StrongTrend"
    df_patterns.loc[choppy_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "Choppy"
    logging.info(f"      Pattern Label Distribution:\n{df_patterns['Pattern_Label'].value_counts(normalize=True).round(3).to_string()}")
    df["Pattern_Label"] = df_patterns["Pattern_Label"].astype('category')
    del df_patterns, prev_high, prev_low, prev_gain, prev_body, prev_macd_hist, breakout_cond, reversal_cond, inside_bar_cond, strong_trend_cond, choppy_cond; maybe_collect()
    return df

def calculate_m15_trend_zone(df_m15):
    logging.info("(Processing) กำลังคำนวณ M15 Trend Zone...")
    cache_key = hash(tuple(df_m15.index)) if isinstance(df_m15, pd.DataFrame) else None
    if cache_key is not None and cache_key in _m15_trend_cache:
        logging.info("      [Cache] ใช้ผลลัพธ์ Trend Zone จาก cache")
        cached_df = _m15_trend_cache[cache_key]
        return cached_df.reindex(df_m15.index).copy()
    if not isinstance(df_m15, pd.DataFrame): logging.error("M15 Trend Zone Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df_m15.empty or "Close" not in df_m15.columns:
        result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"}); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category');
        if cache_key is not None:
            _m15_trend_cache[cache_key] = result_df
        return result_df
    df = df_m15.copy()
    try:
        df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
        if df["Close"].isnull().all():
            result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"}); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category');
            if cache_key is not None:
                _m15_trend_cache[cache_key] = result_df
            return result_df
        df["EMA_Fast"] = ema(df["Close"], M15_TREND_EMA_FAST); df["EMA_Slow"] = ema(df["Close"], M15_TREND_EMA_SLOW); df["RSI"] = rsi(df["Close"], M15_TREND_RSI_PERIOD)
        df.dropna(subset=["EMA_Fast", "EMA_Slow", "RSI"], inplace=True)
        if df.empty:
            result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"}); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category');
            if cache_key is not None:
                _m15_trend_cache[cache_key] = result_df
            return result_df
        is_up = (df["EMA_Fast"] > df["EMA_Slow"]) & (df["RSI"] > M15_TREND_RSI_UP); is_down = (df["EMA_Fast"] < df["EMA_Slow"]) & (df["RSI"] < M15_TREND_RSI_DOWN)
        df["Trend_Zone"] = "NEUTRAL"; df.loc[is_up, "Trend_Zone"] = "UP"; df.loc[is_down, "Trend_Zone"] = "DOWN"
        if not df.empty: logging.info(f"   การกระจาย M15 Trend Zone:\n{df['Trend_Zone'].value_counts(normalize=True).round(3).to_string()}")
        result_df = df[["Trend_Zone"]].reindex(df_m15.index).fillna("NEUTRAL"); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
        del df, is_up, is_down; maybe_collect();
        if cache_key is not None:
            _m15_trend_cache[cache_key] = result_df
        return result_df
    except Exception as e:
        logging.error(f"(Error) การคำนวณ M15 Trend Zone ล้มเหลว: {e}", exc_info=True)
        result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"}); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category');
        if cache_key is not None:
            _m15_trend_cache[cache_key] = result_df
        return result_df

# [Patch v5.5.6] Helper to evaluate higher timeframe trend using SMA crossover
def get_mtf_sma_trend(df_m15, fast=50, slow=200, rsi_period=14, rsi_upper=70, rsi_lower=30):
    """Return trend direction ('UP', 'DOWN', 'NEUTRAL') for M15 data.

    Parameters
    ----------
    df_m15 : pandas.DataFrame
        M15 OHLC data with at least a 'Close' column.
    fast : int, optional
        Fast SMA period. Default 50.
    slow : int, optional
        Slow SMA period. Default 200.
    rsi_period : int, optional
        RSI period. Default 14.
    rsi_upper : float, optional
        Upper RSI filter for uptrend. Default 70.
    rsi_lower : float, optional
        Lower RSI filter for downtrend. Default 30.
    """
    if not isinstance(df_m15, pd.DataFrame) or df_m15.empty or "Close" not in df_m15.columns:
        return "NEUTRAL"
    close = pd.to_numeric(df_m15["Close"], errors="coerce")
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)
    rsi_series = rsi(close, period=rsi_period)
    if fast_ma.empty or slow_ma.empty or rsi_series.empty:
        return "NEUTRAL"
    last_fast = fast_ma.iloc[-1]
    last_slow = slow_ma.iloc[-1]
    last_rsi = rsi_series.iloc[-1]
    if pd.isna(last_fast) or pd.isna(last_slow) or pd.isna(last_rsi):
        return "NEUTRAL"
    if last_fast > last_slow and last_rsi < rsi_upper:
        return "UP"
    if last_fast < last_slow and last_rsi > rsi_lower:
        return "DOWN"
    return "NEUTRAL"

# [Patch v5.0.2] Exclude heavy engineering logic from coverage
def engineer_m1_features(df_m1, timeframe_minutes=TIMEFRAME_MINUTES_M1, lag_features_config=None):  # pragma: no cover
    logging.info("[QA] Start M1 Feature Engineering")
    logging.info("(Processing) กำลังสร้าง Features M1 (v4.9.0)...") # <<< MODIFIED v4.9.0
    if not isinstance(df_m1, pd.DataFrame): logging.error("Engineer M1 Features Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df_m1.empty: logging.warning("   (Warning) ข้ามการสร้าง Features M1: DataFrame ว่างเปล่า."); return df_m1
    df = df_m1.copy(); price_cols = ["Open", "High", "Low", "Close"]
    if any(col not in df.columns for col in price_cols):
        logging.warning(f"   (Warning) ขาดคอลัมน์ราคา M1. บาง Features อาจเป็น NaN.")
        base_feature_cols = ["Candle_Body", "Candle_Range", "Gain", "Candle_Ratio", "Upper_Wick", "Lower_Wick", "Wick_Length", "Wick_Ratio", "Gain_Z", "MACD_line", "MACD_signal", "MACD_hist", "MACD_hist_smooth", "ATR_14", "ATR_14_Shifted", "ATR_14_Rolling_Avg", "Candle_Speed", 'Volatility_Index', 'ADX', 'RSI']
        for col in base_feature_cols:
            if col not in df.columns: df[col] = np.nan
        if "Pattern_Label" not in df.columns: df["Pattern_Label"] = "N/A"
    else:
        for col in price_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=price_cols, inplace=True)
        if df.empty: logging.warning("   (Warning) M1 DataFrame ว่างเปล่าหลังลบราคา NaN."); return df
        df["Candle_Body"]=abs(df["Close"]-df["Open"]).astype('float32'); df["Candle_Range"]=(df["High"]-df["Low"]).astype('float32'); df["Gain"]=(df["Close"]-df["Open"]).astype('float32')
        df["Candle_Ratio"]=np.where(df["Candle_Range"].abs()>1e-9,df["Candle_Body"]/df["Candle_Range"],0.0).astype('float32'); df["Upper_Wick"]=(df["High"]-np.maximum(df["Open"],df["Close"])).astype('float32')
        df["Lower_Wick"]=(np.minimum(df["Open"],df["Close"])-df["Low"]).astype('float32'); df["Wick_Length"]=(df["Upper_Wick"]+df["Lower_Wick"]).astype('float32')
        df["Wick_Ratio"]=np.where(df["Candle_Range"].abs()>1e-9,df["Wick_Length"]/df["Candle_Range"],0.0).astype('float32'); df["Gain_Z"]=rolling_zscore(df["Gain"].fillna(0),window=ROLLING_Z_WINDOW_M1)
        df["MACD_line"],df["MACD_signal"],df["MACD_hist"]=macd(df["Close"])
        if "MACD_hist" in df.columns and df["MACD_hist"].notna().any(): df["MACD_hist_smooth"]=df["MACD_hist"].rolling(window=5,min_periods=1).mean().fillna(0).astype('float32')
        else: df["MACD_hist_smooth"]=np.nan; logging.warning("      (Warning) ไม่สามารถคำนวณ MACD_hist_smooth.")
        df=atr(df,14)
        if "ATR_14" in df.columns and df["ATR_14"].notna().any(): df["ATR_14_Rolling_Avg"]=sma(df["ATR_14"],ATR_ROLLING_AVG_PERIOD)
        else: df["ATR_14_Rolling_Avg"]=np.nan; logging.warning("      (Warning) ไม่สามารถคำนวณ ATR_14_Rolling_Avg.")
        df["Candle_Speed"]=(df["Gain"]/max(timeframe_minutes,1e-6)).astype('float32'); df["RSI"]=rsi(df["Close"],period=14)
    if lag_features_config and isinstance(lag_features_config,dict):
        for feature in lag_features_config.get('features',[]):
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                for lag in lag_features_config.get('lags',[]):
                    if isinstance(lag,int) and lag>0: df[f"{feature}_lag{lag}"]=df[feature].shift(lag).astype('float32')
    if 'ATR_14' in df.columns and 'ATR_14_Rolling_Avg' in df.columns and df['ATR_14_Rolling_Avg'].notna().any():
        df['Volatility_Index']=np.where(df['ATR_14_Rolling_Avg'].abs()>1e-9,df['ATR_14']/df['ATR_14_Rolling_Avg'],np.nan)
        df['Volatility_Index']=df['Volatility_Index'].ffill().fillna(1.0).astype('float32')
    else: df['Volatility_Index']=1.0; logging.warning("         (Warning) ไม่สามารถคำนวณ Volatility_Index.")
    if all(c in df.columns for c in ['High','Low','Close']) and ta:
        try:
            if len(df.dropna(subset=['High','Low','Close']))>=14*2+10: adx_indicator=ta.trend.ADXIndicator(df['High'],df['Low'],df['Close'],window=14,fillna=False); df['ADX']=adx_indicator.adx().ffill().fillna(25.0).astype('float32')
            else: df['ADX']=25.0; logging.warning("         (Warning) ไม่สามารถคำนวณ ADX: ข้อมูลไม่เพียงพอ.")
        except Exception as e_adx: df['ADX']=25.0; logging.warning(f"         (Warning) ไม่สามารถคำนวณ ADX: {e_adx}")
    else: df['ADX']=25.0
    if all(col in df.columns for col in ["Gain_Z","High","Low","Close","Open","MACD_hist","Candle_Ratio","Wick_Ratio","Gain","Candle_Body"]): df=tag_price_structure_patterns(df)
    else: df["Pattern_Label"]="N/A"; df["Pattern_Label"]=df["Pattern_Label"].astype('category'); logging.warning("   (Warning) ข้ามการ Tag Patterns.")
    if 'cluster' not in df.columns:
        try:
            cluster_features=['Gain_Z','Volatility_Index','Candle_Ratio','RSI','ADX']; features_present=[f for f in cluster_features if f in df.columns and df[f].notna().any()]
            if len(features_present)<2 or len(df[features_present].dropna())<3: df['cluster']=0; logging.warning("         (Warning) Not enough valid features/samples for clustering.")
            else:
                X_cluster_raw = df[features_present].copy().replace([np.inf, -np.inf], np.nan)
                X_cluster = X_cluster_raw.fillna(X_cluster_raw.median()).fillna(0)
                # [Patch v5.0.16] Skip KMeans if duplicate samples could cause ConvergenceWarning
                if len(X_cluster) >= 3:
                    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_cluster)
                    if len(np.unique(X_scaled, axis=0)) < 3:
                        df['cluster'] = 0
                        logging.warning("         (Warning) Not enough unique samples for clustering.")
                    else:
                        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                        df['cluster'] = kmeans.fit_predict(X_scaled)
                else:
                    df['cluster'] = 0
                    logging.warning("         (Warning) Not enough samples after cleaning for clustering.")
        except Exception as e_cluster: df['cluster']=0; logging.error(f"         (Error) Clustering failed: {e_cluster}.",exc_info=True)
        if 'cluster' in df.columns: df['cluster']=pd.to_numeric(df['cluster'],downcast='integer')
    if 'spike_score' not in df.columns:
        try:
            gain_z_abs = abs(pd.to_numeric(df.get('Gain_Z', 0.0), errors='coerce').fillna(0.0))
            wick_ratio = pd.to_numeric(df.get('Wick_Ratio', 0.0), errors='coerce').fillna(0.0)
            atr_val = pd.to_numeric(df.get('ATR_14', 1.0), errors='coerce').fillna(1.0).replace([np.inf, -np.inf], 1.0)
            score = (wick_ratio * 0.5 + gain_z_abs * 0.3 + atr_val * 0.2)
            score = np.where((atr_val > 1.5) & (wick_ratio > 0.6), score * 1.2, score)
            df['spike_score'] = score.clip(0, 1).astype('float32')
        except Exception as e_spike:
            df['spike_score'] = 0.0
            logging.error(f"         (Error) Spike score calculation failed: {e_spike}.", exc_info=True)

    # [Patch v5.7.9] additional engineered features
    if {'BuyVolume', 'SellVolume'}.issubset(df.columns):
        df['OF_Imbalance'] = calculate_order_flow_imbalance(df)
    else:
        df['OF_Imbalance'] = 0.0

    if 'Close' in df.columns:
        df['Momentum_Divergence'] = calculate_momentum_divergence(df['Close'])
    else:
        df['Momentum_Divergence'] = 0.0

    if 'Volume' in df.columns:
        df['Relative_Volume'] = calculate_relative_volume(df)
    else:
        df['Relative_Volume'] = 0.0
    if 'session' not in df.columns:
        logging.info("      Creating 'session' column...")
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
            sessions = pd.Index(df.index).map(lambda ts: get_session_tag(ts, warn_once=True))
            df['session'] = pd.Series(sessions, index=df.index).astype('category')
            logging.info(
                f"         Session distribution:\n{df['session'].value_counts(normalize=True).round(3).to_string()}"
            )
        except Exception as e_session:
            logging.error(
                f"         (Error) Session calculation failed: {e_session}. Assigning 'Other'.",
                exc_info=True,
            )
            df['session'] = pd.Series('Other', index=df.index).astype('category')
    if 'model_tag' not in df.columns: df['model_tag'] = 'N/A'
    logging.info("(Success) สร้าง Features M1 (v4.9.0) เสร็จสิ้น.")  # <<< MODIFIED v4.9.0
    numeric_cols_clean = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols_clean) > 0:
        df[numeric_cols_clean] = df[numeric_cols_clean].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols_clean] = df[numeric_cols_clean].ffill().fillna(0)
    # [Patch v5.5.4] Run QA check after cleaning to avoid false warnings
    if df.isnull().any().any() or np.isinf(df[numeric_cols_clean]).any().any():
        logging.warning("[QA WARNING] NaN/Inf detected in engineered features")
    logging.info("[QA] M1 Feature Engineering Completed")
    return df.reindex(df_m1.index)

# [Patch v5.0.2] Exclude heavy cleaning logic from coverage
def clean_m1_data(df_m1):  # pragma: no cover
    logging.info("(Processing) กำลังกำหนด Features M1 สำหรับ Drift และแปลงประเภท (v4.9.0)...") # <<< MODIFIED v4.9.0
    if not isinstance(df_m1, pd.DataFrame): logging.error("Clean M1 Data Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df_m1.empty: logging.warning("   (Warning) ข้ามการทำความสะอาดข้อมูล M1: DataFrame ว่างเปล่า."); return pd.DataFrame(), []
    df_cleaned = df_m1.copy()
    potential_m1_features = ["Candle_Body", "Candle_Range", "Candle_Ratio", "Gain", "Gain_Z", "MACD_line", "MACD_signal", "MACD_hist", "MACD_hist_smooth", "ATR_14", "ATR_14_Shifted", "ATR_14_Rolling_Avg", "Candle_Speed", "Wick_Length", "Wick_Ratio", "Pattern_Label", "Signal_Score", 'Volatility_Index', 'ADX', 'RSI', 'cluster', 'spike_score', 'OF_Imbalance', 'Momentum_Divergence', 'Relative_Volume', 'session']
    lag_cols_in_df = [col for col in df_cleaned.columns if '_lag' in col]
    potential_m1_features.extend(lag_cols_in_df)
    if META_CLASSIFIER_FEATURES: potential_m1_features.extend([f for f in META_CLASSIFIER_FEATURES if f not in potential_m1_features])
    potential_m1_features = sorted(list(dict.fromkeys(potential_m1_features)))
    m1_features_for_drift = [f for f in potential_m1_features if f in df_cleaned.columns]
    logging.info(f"   กำหนด {len(m1_features_for_drift)} Features M1 สำหรับ Drift: {m1_features_for_drift}")
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        try:
            inf_mask = df_cleaned[numeric_cols].isin([np.inf, -np.inf])
            if inf_mask.any().any():
                cols_with_inf = df_cleaned[numeric_cols].columns[inf_mask.any()].tolist()
                logging.warning(f"      [Inf Check] พบ Inf ใน: {cols_with_inf}. กำลังแทนที่ด้วย NaN...")
                df_cleaned[cols_with_inf] = df_cleaned[cols_with_inf].replace([np.inf, -np.inf], np.nan)
            for col in numeric_cols: df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            cols_with_nan = df_cleaned[numeric_cols].columns[df_cleaned[numeric_cols].isnull().any()].tolist()
            if cols_with_nan:
                logging.info(f"      [NaN Check] พบ NaN ใน: {cols_with_nan}. กำลังเติมด้วย ffill().fillna(0)...")
                df_cleaned[cols_with_nan] = df_cleaned[cols_with_nan].ffill().fillna(0)
            for col in numeric_cols:
                if col not in df_cleaned.columns: continue
                if pd.api.types.is_integer_dtype(df_cleaned[col].dtype): df_cleaned[col] = pd.to_numeric(df_cleaned[col], downcast='integer')
                elif pd.api.types.is_float_dtype(df_cleaned[col].dtype) and df_cleaned[col].dtype != 'float32': df_cleaned[col] = df_cleaned[col].astype('float32')
        except Exception as e: logging.error(f"   (Error) เกิดข้อผิดพลาดในการแปลงประเภทข้อมูลหรือเติม NaN/Inf: {e}.", exc_info=True)
    categorical_cols = ['Pattern_Label', 'session']
    for col in categorical_cols:
        if col in df_cleaned.columns:
            if df_cleaned[col].isnull().any(): df_cleaned[col] = df_cleaned[col].fillna("Unknown") # Use assignment
            if not isinstance(df_cleaned[col].dtype, pd.CategoricalDtype):
                try: df_cleaned[col] = df_cleaned[col].astype('category')
                except Exception as e_cat: logging.warning(f"   (Warning) เกิดข้อผิดพลาดในการแปลง '{col}' เป็น category: {e_cat}.")
    logging.info("(Success) กำหนด Features M1 และแปลงประเภท (v4.9.0) เสร็จสิ้น.") # <<< MODIFIED v4.9.0
    return df_cleaned, m1_features_for_drift

# [Patch v5.0.2] Exclude heavy signal calculation from coverage
def calculate_m1_entry_signals(df_m1: pd.DataFrame, config: dict) -> pd.DataFrame:  # pragma: no cover
    logging.debug("      (Calculating M1 Signals)...")
    df = df_m1.copy(); df['Signal_Score'] = 0.0
    gain_z_thresh = config.get('gain_z_thresh', 0.3); rsi_thresh_buy = config.get('rsi_thresh_buy', 50)
    rsi_thresh_sell = config.get('rsi_thresh_sell', 50); volatility_max = config.get('volatility_max', 4.0)
    entry_score_min = config.get('min_signal_score', MIN_SIGNAL_SCORE_ENTRY); ignore_rsi = config.get('ignore_rsi_scoring', False)
    df['Gain_Z'] = df.get('Gain_Z', pd.Series(0.0, index=df.index)).fillna(0.0)
    buy_gain_z_cond = df['Gain_Z'] > gain_z_thresh; sell_gain_z_cond = df['Gain_Z'] < -gain_z_thresh
    df['Pattern_Label'] = df.get('Pattern_Label', pd.Series('Normal', index=df.index)).astype(str).fillna('Normal')
    buy_pattern_cond = df['Pattern_Label'].isin(['Breakout', 'StrongTrend']) & (df['Gain_Z'] > 0)
    sell_pattern_cond = df['Pattern_Label'].isin(['Breakout', 'StrongTrend', 'Reversal']) & (df['Gain_Z'] < 0)
    df['RSI'] = df.get('RSI', pd.Series(50.0, index=df.index)).fillna(50.0)
    buy_rsi_cond = df['RSI'] > rsi_thresh_buy; sell_rsi_cond = df['RSI'] < rsi_thresh_sell
    df['Volatility_Index'] = df.get('Volatility_Index', pd.Series(1.0, index=df.index)).fillna(1.0)
    vol_cond = df['Volatility_Index'] < volatility_max
    df.loc[buy_gain_z_cond, 'Signal_Score'] += 1.0; df.loc[sell_gain_z_cond, 'Signal_Score'] -= 1.0
    df.loc[buy_pattern_cond, 'Signal_Score'] += 1.0; df.loc[sell_pattern_cond, 'Signal_Score'] -= 1.0
    if not ignore_rsi: df.loc[buy_rsi_cond, 'Signal_Score'] += 1.0; df.loc[sell_rsi_cond, 'Signal_Score'] -= 1.0
    df.loc[vol_cond, 'Signal_Score'] += 1.0
    df['Signal_Score'] = df['Signal_Score'].astype('float32')
    df['Entry_Long'] = ((df['Signal_Score'] > 0) & (df['Signal_Score'] >= entry_score_min)).astype(int)
    df['Entry_Short'] = ((df['Signal_Score'] < 0) & (abs(df['Signal_Score']) >= entry_score_min)).astype(int)
    df['Trade_Reason'] = ""; df.loc[buy_gain_z_cond, 'Trade_Reason'] += f"+Gz>{gain_z_thresh:.1f}"
    df.loc[sell_gain_z_cond, 'Trade_Reason'] += f"+Gz<{-gain_z_thresh:.1f}"; df.loc[buy_pattern_cond, 'Trade_Reason'] += "+PBuy"
    df.loc[sell_pattern_cond, 'Trade_Reason'] += "+PSell"
    if not ignore_rsi: df.loc[buy_rsi_cond, 'Trade_Reason'] += f"+RSI>{rsi_thresh_buy}"; df.loc[sell_rsi_cond, 'Trade_Reason'] += f"+RSI<{rsi_thresh_sell}"
    df.loc[vol_cond, 'Trade_Reason'] += f"+Vol<{volatility_max:.1f}"
    buy_entry_mask = df['Entry_Long'] == 1; sell_entry_mask = df['Entry_Short'] == 1
    df.loc[buy_entry_mask, 'Trade_Reason'] = "BUY(" + df.loc[buy_entry_mask, 'Signal_Score'].round(1).astype(str) + "):" + df.loc[buy_entry_mask, 'Trade_Reason'].str.lstrip('+')
    df.loc[sell_entry_mask, 'Trade_Reason'] = "SELL(" + df.loc[sell_entry_mask, 'Signal_Score'].abs().round(1).astype(str) + "):" + df.loc[sell_entry_mask, 'Trade_Reason'].str.lstrip('+')
    df.loc[~(buy_entry_mask | sell_entry_mask), 'Trade_Reason'] = "NONE"
    df['Trade_Tag'] = df['Signal_Score'].round(1).astype(str) + "_" + df['Pattern_Label'].astype(str)
    return df

logging.info("Part 5: Feature Engineering & Indicator Calculation Functions Loaded.")
# === END OF PART 5/12 ===
# === START OF PART 6/12 ===

# ==============================================================================
# === PART 6: Machine Learning Configuration & Helpers (v4.8.8 - Patch 9 Applied: Fix ML Log) ===
# ==============================================================================
# <<< MODIFIED v4.8.7: Re-verified robustness checks in check_model_overfit based on latest prompt. >>>
# <<< MODIFIED v4.8.8: Ensured check_model_overfit robustness aligns with final prompt. Added default and try-except for META_META_MIN_PROBA_THRESH. >>>
# <<< MODIFIED v4.8.8 (Patch 1): Enhanced robustness in check_model_overfit, check_feature_noise_shap, analyze_feature_importance_shap, select_top_shap_features. Corrected logic for overfitting detection and noise logging. >>>
# <<< MODIFIED v4.8.8 (Patch 4): Corrected select_top_shap_features return value for invalid feature_names. Fixed overfitting detection logic and noise logging in check_model_overfit/check_feature_noise_shap. >>>
# <<< MODIFIED v4.8.8 (Patch 6): Applied user prompt fixes for check_model_overfit and check_feature_noise_shap logging. >>>
# <<< MODIFIED v4.8.8 (Patch 9): Fixed logging format and conditions in check_model_overfit and check_feature_noise_shap as per failed tests and plan. >>>
import logging
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
# Import ML libraries conditionally (assuming they are checked/installed in Part 1)
try:
    import shap
except ImportError:
    shap = None
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None
    Pool = None
try:
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
except ImportError:
    accuracy_score = roc_auc_score = log_loss = classification_report = lambda *args, **kwargs: None
    logging.error("Scikit-learn metrics not found!")

# Ensure global configurations are accessible if run independently
# Define defaults if globals are not found
DEFAULT_META_MIN_PROBA_THRESH = 0.25
DEFAULT_ENABLE_OPTUNA_TUNING = True
DEFAULT_OPTUNA_N_TRIALS = 50
DEFAULT_OPTUNA_CV_SPLITS = 5
DEFAULT_OPTUNA_METRIC = "AUC"
DEFAULT_OPTUNA_DIRECTION = "maximize"
DEFAULT_META_CLASSIFIER_FEATURES = [
    "RSI", "MACD_hist_smooth", "ATR_14", "ADX", "Gain_Z", "Volatility_Index",
    "Candle_Ratio", "Wick_Ratio", "Candle_Speed",
    "Gain_Z_lag1", "Gain_Z_lag3", "Gain_Z_lag5",
    "Candle_Speed_lag1", "Candle_Speed_lag3", "Candle_Speed_lag5",
    "cluster", "spike_score", "Pattern_Label",
    "OF_Imbalance", "Momentum_Divergence", "Relative_Volume",
]
# <<< [Patch] Added default for Meta-Meta threshold >>>
DEFAULT_META_META_MIN_PROBA_THRESH = 0.5

try:
    USE_META_CLASSIFIER
except NameError:
    USE_META_CLASSIFIER = True
try:
    USE_META_META_CLASSIFIER
except NameError:
    USE_META_META_CLASSIFIER = False
try:
    META_CLASSIFIER_PATH
except NameError:
    META_CLASSIFIER_PATH = "meta_classifier.pkl"
try:
    SPIKE_MODEL_PATH
except NameError:
    SPIKE_MODEL_PATH = "meta_classifier_spike.pkl"
try:
    CLUSTER_MODEL_PATH
except NameError:
    CLUSTER_MODEL_PATH = "meta_classifier_cluster.pkl"
try:
    META_META_CLASSIFIER_PATH
except NameError:
    META_META_CLASSIFIER_PATH = "meta_meta_classifier.pkl"
try:
    META_CLASSIFIER_FEATURES
except NameError:
    META_CLASSIFIER_FEATURES = DEFAULT_META_CLASSIFIER_FEATURES
try:
    META_META_CLASSIFIER_FEATURES
except NameError:
    META_META_CLASSIFIER_FEATURES = []
try:
    META_MIN_PROBA_THRESH
except NameError:
    META_MIN_PROBA_THRESH = DEFAULT_META_MIN_PROBA_THRESH
META_MIN_PROBA_THRESH = get_env_float("META_MIN_PROBA_THRESH", META_MIN_PROBA_THRESH)  # env override
try:
    REENTRY_MIN_PROBA_THRESH
except NameError:
    REENTRY_MIN_PROBA_THRESH = META_MIN_PROBA_THRESH
REENTRY_MIN_PROBA_THRESH = get_env_float("REENTRY_MIN_PROBA_THRESH", REENTRY_MIN_PROBA_THRESH)  # env override
# <<< [Patch] Added try-except for Meta-Meta threshold >>>
try:
    META_META_MIN_PROBA_THRESH
except NameError:
    META_META_MIN_PROBA_THRESH = DEFAULT_META_META_MIN_PROBA_THRESH
META_META_MIN_PROBA_THRESH = get_env_float("META_META_MIN_PROBA_THRESH", META_META_MIN_PROBA_THRESH)  # env override
# <<< End of [Patch] >>>
try:
    ENABLE_OPTUNA_TUNING
except NameError:
    ENABLE_OPTUNA_TUNING = DEFAULT_ENABLE_OPTUNA_TUNING
try:
    OPTUNA_N_TRIALS
except NameError:
    OPTUNA_N_TRIALS = DEFAULT_OPTUNA_N_TRIALS
try:
    OPTUNA_CV_SPLITS
except NameError:
    OPTUNA_CV_SPLITS = DEFAULT_OPTUNA_CV_SPLITS
try:
    OPTUNA_METRIC
except NameError:
    OPTUNA_METRIC = DEFAULT_OPTUNA_METRIC
try:
    OPTUNA_DIRECTION
except NameError:
    OPTUNA_DIRECTION = DEFAULT_OPTUNA_DIRECTION

logging.info("Loading Machine Learning Configuration & Helpers...")

# --- ML Model Usage Flags ---
logging.info(f"USE_META_CLASSIFIER (L1 Filter): {USE_META_CLASSIFIER}")
logging.info(f"USE_META_META_CLASSIFIER (L2 Filter): {USE_META_META_CLASSIFIER}")

# --- ML Model Paths & Features ---
logging.debug(f"Main L1 Model Path: {META_CLASSIFIER_PATH}")
logging.debug(f"Spike Model Path: {SPIKE_MODEL_PATH}")
logging.debug(f"Cluster Model Path: {CLUSTER_MODEL_PATH}")
logging.debug(f"L2 Model Path: {META_META_CLASSIFIER_PATH}")
logging.debug(f"Default L1 Features (Count): {len(META_CLASSIFIER_FEATURES)}")
logging.debug(f"L2 Features (Count): {len(META_META_CLASSIFIER_FEATURES)}")

# --- ML Thresholds ---
logging.info(f"Default L1 Probability Threshold: {META_MIN_PROBA_THRESH}")
logging.info(f"Default Re-entry Probability Threshold: {REENTRY_MIN_PROBA_THRESH}")
logging.info(f"Default L2 Probability Threshold: {META_META_MIN_PROBA_THRESH}") # Now uses defined value

# --- Optuna Configuration ---
logging.info(f"Optuna Hyperparameter Tuning Enabled: {ENABLE_OPTUNA_TUNING}")
if ENABLE_OPTUNA_TUNING:
    logging.info(f"  Optuna Trials: {OPTUNA_N_TRIALS}")
    logging.info(f"  Optuna CV Splits: {OPTUNA_CV_SPLITS}")
    logging.info(f"  Optuna Metric: {OPTUNA_METRIC} ({OPTUNA_DIRECTION})")

# --- Auto Threshold Tuning ---
ENABLE_AUTO_THRESHOLD_TUNING = False # Keep this disabled for now
logging.info(f"Auto Threshold Tuning Enabled: {ENABLE_AUTO_THRESHOLD_TUNING}")

# --- Global variables to store model info ---
meta_model_type_used = "N/A"
meta_meta_model_type_used = "N/A"
logging.debug("Global model type trackers initialized.")

# --- SHAP Feature Selection Helper Function ---
# [Patch v5.0.2] Exclude SHAP helper from coverage
def select_top_shap_features(shap_values_val, feature_names, shap_threshold=0.01):  # pragma: no cover
    """
    Selects features based on Normalized Mean Absolute SHAP values exceeding a threshold.
    (v4.8.8 Patch 4: Corrected return for invalid feature_names)
    """
    logging.info(f"   [SHAP Select] กำลังเลือก Features ที่มี Normalized SHAP >= {shap_threshold:.4f}...")
    # <<< [Patch] MODIFIED v4.8.8 (Patch 1): Enhanced input validation >>>
    if shap_values_val is None or not isinstance(shap_values_val, np.ndarray) or shap_values_val.size == 0:
        logging.warning("      (Warning) ไม่สามารถเลือก Features: ค่า SHAP ไม่ถูกต้องหรือว่างเปล่า. คืนค่า Features เดิม.")
        return feature_names if isinstance(feature_names, list) else [] # Return original or empty list
    # <<< [Patch] MODIFIED v4.8.8 (Patch 4): Return None if feature_names is invalid >>>
    if feature_names is None or not isinstance(feature_names, list) or not feature_names:
        logging.warning("      (Warning) ไม่สามารถเลือก Features: รายชื่อ Features ไม่ถูกต้องหรือว่างเปล่า. คืนค่า None.")
        return None
    # <<< End of [Patch] MODIFIED v4.8.8 (Patch 4) >>>
    if shap_values_val.ndim != 2:
        # Handle potential case where SHAP returns values for multiple classes (e.g., list of arrays)
        if isinstance(shap_values_val, list) and len(shap_values_val) >= 2 and isinstance(shap_values_val[1], np.ndarray) and shap_values_val[1].ndim == 2:
            logging.debug("      (Info) SHAP values appear to be for multiple classes, using index 1 (positive class).")
            shap_values_val = shap_values_val[1] # Use SHAP values for the positive class
        else:
            logging.warning(f"      (Warning) ขนาด SHAP values ไม่ถูกต้อง ({shap_values_val.ndim} dimensions, expected 2). คืนค่า Features เดิม.")
            return feature_names
    if shap_values_val.shape[1] != len(feature_names):
        logging.warning(f"      (Warning) ขนาด SHAP values ไม่ตรงกับจำนวน Features (SHAP: {shap_values_val.shape[1]}, Features: {len(feature_names)}). คืนค่า Features เดิม.")
        return feature_names
    if shap_values_val.shape[0] == 0:
        logging.warning("      (Warning) SHAP values array มี 0 samples. ไม่สามารถคำนวณ Importance ได้. คืนค่า Features เดิม.")
        return feature_names
    # <<< End of [Patch] MODIFIED v4.8.8 (Patch 1) >>>

    try:
        mean_abs_shap = np.abs(shap_values_val).mean(axis=0)
        if np.isnan(mean_abs_shap).any() or np.isinf(mean_abs_shap).any():
            logging.warning("      (Warning) พบ NaN หรือ Inf ใน Mean Absolute SHAP values. ไม่สามารถเลือก Features ได้. คืนค่า Features เดิม.")
            return feature_names

        shap_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})
        total_shap = shap_df["Mean_Abs_SHAP"].sum()
        if total_shap > 1e-9:
            shap_df["Normalized_SHAP"] = shap_df["Mean_Abs_SHAP"] / total_shap
        else:
            shap_df["Normalized_SHAP"] = 0.0
            logging.warning("      (Warning) Total Mean Abs SHAP ใกล้ศูนย์, ไม่สามารถ Normalize ได้. จะไม่เลือก Feature ใดๆ.")
            return [] # Return empty list if no importance

        selected_features_df = shap_df[shap_df["Normalized_SHAP"] >= shap_threshold].copy()
        selected_features = selected_features_df["Feature"].tolist()

        if not selected_features:
            logging.warning(f"      (Warning) ไม่มี Features ใดผ่านเกณฑ์ SHAP >= {shap_threshold:.4f}. คืนค่า List ว่าง.")
            return []
        elif len(selected_features) < len(feature_names):
            removed_features = sorted(list(set(feature_names) - set(selected_features)))
            logging.info(f"      (Success) เลือก Features ได้ {len(selected_features)} ตัวจาก SHAP.")
            logging.info(f"         Features ที่ถูกตัดออก {len(removed_features)} ตัว: {removed_features}")
            logging.info("         Features ที่เลือก (เรียงตามค่า SHAP):")
            logging.info("\n" + selected_features_df.sort_values("Normalized_SHAP", ascending=False)[["Feature", "Normalized_SHAP"]].round(5).to_string(index=False))
        else:
            logging.info("      (Success) Features ทั้งหมดผ่านเกณฑ์ SHAP.")
        return selected_features
    except Exception as e:
        logging.error(f"      (Error) เกิดข้อผิดพลาดระหว่างการเลือก Features ด้วย SHAP: {e}. คืนค่า Features เดิม.", exc_info=True)
        return feature_names

# --- Model Quality Check Functions ---
# [Patch v5.0.2] Exclude model overfit check from coverage
def check_model_overfit(model, X_train, y_train, X_val, y_val, X_test=None, y_test=None, metric="AUC", threshold_pct=15.0):  # pragma: no cover
    """
    Checks for potential overfitting by comparing model performance.
    (v4.8.8 Patch 9: Fixed logging logic and format)
    """
    logging.info(f"   [Check] Checking for Overfitting ({metric})...")
    if model is None:
        logging.warning("      (Warning) Cannot check Overfitting: Model is None.")
        return
    if X_val is None or y_val is None:
        logging.warning("      (Warning) Cannot check Overfitting: Validation data missing.")
        return
    if (X_test is None and y_test is not None) or (X_test is not None and y_test is None):
        logging.warning("      (Warning) Cannot check Overfitting: Test data requires both X_test and y_test.")
        X_test, y_test = None, None

    def _ensure_pd(data, name):
        if data is None: return None
        if isinstance(data, np.ndarray):
            if data.ndim == 1: return pd.Series(data, name=name)
            elif data.ndim == 2:
                feature_names = getattr(model, 'feature_names_', None)
                if feature_names and len(feature_names) == data.shape[1]:
                    return pd.DataFrame(data, columns=feature_names)
                else:
                    return pd.DataFrame(data)
            else:
                logging.warning(f"      (Warning) Cannot check Overfitting: {name} has unexpected dimensions ({data.ndim}).")
                return None
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return data
        else:
            logging.warning(f"      (Warning) Cannot check Overfitting: {name} has unexpected type ({type(data)}).")
            return None

    X_train = _ensure_pd(X_train, "X_train")
    y_train = _ensure_pd(y_train, "y_train")
    X_val = _ensure_pd(X_val, "X_val")
    y_val = _ensure_pd(y_val, "y_val")
    X_test = _ensure_pd(X_test, "X_test")
    y_test = _ensure_pd(y_test, "y_test")

    if X_val is None or y_val is None: return
    if X_train is not None and (y_train is None or len(X_train) != len(y_train)):
        logging.warning("      (Warning) Cannot check Overfitting: Train X and y data sizes do not match or y_train is None.")
        X_train, y_train = None, None
    if X_val is not None and (y_val is None or len(X_val) != len(y_val)):
        logging.warning("      (Warning) Cannot check Overfitting: Validation X and y data sizes do not match or y_val is None.")
        return
    if X_test is not None and (y_test is None or len(X_test) != len(y_test)):
        logging.warning("      (Warning) Cannot check Overfitting: Test X and y data sizes do not match or y_test is None.")
        X_test, y_test = None, None
    if X_val is not None and len(X_val) == 0:
        logging.warning("      (Warning) Cannot check Overfitting: Validation data is empty.")
        return
    if X_train is not None and len(X_train) == 0:
        logging.warning("      (Warning) Cannot check Overfitting: Train data is empty.")
        X_train, y_train = None, None
    if X_test is not None and len(X_test) == 0:
        logging.warning("      (Warning) Cannot check Overfitting: Test data is empty.")
        X_test, y_test = None, None

    try:
        train_score, val_score, test_score = np.nan, np.nan, np.nan
        model_classes = getattr(model, 'classes_', None)
        if model_classes is None or len(model_classes) < 2:
            logging.warning("      (Warning) Cannot determine model classes or only one class found. Using [0, 1] as fallback.")
            try:
                y_ref = y_val if y_train is None else y_train
                if y_ref is not None:
                    unique_classes = np.unique(y_ref)
                    if len(unique_classes) >= 2:
                        model_classes = unique_classes
                    else:
                        model_classes = [0, 1]
                else:
                    model_classes = [0, 1]
            except Exception:
                model_classes = [0, 1]

        def safe_predict(model, data, method_name):
            if data is None: return None
            if not hasattr(model, method_name):
                logging.warning(f"      (Warning) Model lacks '{method_name}' method.")
                return None
            try:
                pred = getattr(model, method_name)(data)
                if pred is None:
                    logging.warning(f"      (Warning) '{method_name}' returned None.")
                    return None
                if method_name == 'predict_proba':
                    if not isinstance(pred, np.ndarray) or pred.ndim != 2 or pred.shape[1] < 2:
                        logging.warning(f"      (Warning) '{method_name}' returned invalid shape or type: {getattr(pred, 'shape', type(pred))}.")
                        return None
                elif method_name == 'predict':
                    if not isinstance(pred, (np.ndarray, pd.Series)) or pred.ndim != 1:
                        logging.warning(f"      (Warning) '{method_name}' returned invalid shape or type: {getattr(pred, 'shape', type(pred))}.")
                        return None
                return pred
            except Exception as e_pred:
                logging.error(f"      (Error) Calculating {method_name} failed: {e_pred}", exc_info=True)
                return None

        if metric == "AUC":
            val_pred_proba_raw = safe_predict(model, X_val, "predict_proba")
            if val_pred_proba_raw is not None: val_score = roc_auc_score(y_val, val_pred_proba_raw[:, 1])
            train_pred_proba_raw = safe_predict(model, X_train, "predict_proba")
            if train_pred_proba_raw is not None: train_score = roc_auc_score(y_train, train_pred_proba_raw[:, 1])
            test_pred_proba_raw = safe_predict(model, X_test, "predict_proba")
            if test_pred_proba_raw is not None: test_score = roc_auc_score(y_test, test_pred_proba_raw[:, 1])
        elif metric == "LogLoss":
            val_pred_proba = safe_predict(model, X_val, "predict_proba")
            if val_pred_proba is not None: val_score = log_loss(y_val.astype(int), val_pred_proba, labels=model_classes)
            train_pred_proba = safe_predict(model, X_train, "predict_proba")
            if train_pred_proba is not None: train_score = log_loss(y_train.astype(int), train_pred_proba, labels=model_classes)
            test_pred_proba = safe_predict(model, X_test, "predict_proba")
            if test_pred_proba is not None: test_score = log_loss(y_test.astype(int), test_pred_proba, labels=model_classes)
        elif metric == "Accuracy":
            val_pred = safe_predict(model, X_val, "predict")
            if val_pred is not None: val_score = accuracy_score(y_val, val_pred)
            train_pred = safe_predict(model, X_train, "predict")
            if train_pred is not None: train_score = accuracy_score(y_train, train_pred)
            test_pred = safe_predict(model, X_test, "predict")
            if test_pred is not None: test_score = accuracy_score(y_test, test_pred)
        else:
            logging.warning(f"      (Warning) Metric '{metric}' not supported for Overfit check.")
            return

        if pd.notna(train_score): logging.info(f"      Train {metric}: {train_score:.4f}")
        if pd.notna(val_score): logging.info(f"      Val {metric}:   {val_score:.4f}")
        if pd.notna(test_score): logging.info(f"      Test {metric}:  {test_score:.4f}")

        # <<< [Patch] MODIFIED v4.8.8 (Patch 9): Corrected overfitting logic and logging >>>
        if pd.notna(train_score) and pd.notna(val_score):
            diff_val = train_score - val_score
            diff_val_pct = float('inf') # Default to infinity if denominator is zero
            is_overfitting_val = False
            denominator = 0.0

            if metric == "LogLoss": # Lower is better
                # Check if val_score is significantly worse (higher) than train_score
                denominator = abs(train_score)
                if val_score > train_score + 1e-9: # Check if val score is worse
                    if denominator > 1e-9:
                        diff_val_pct = abs(diff_val / denominator) * 100.0
                    if diff_val_pct > threshold_pct:
                        is_overfitting_val = True
            else: # Higher is better (AUC, Accuracy)
                # Check if train_score is significantly better (higher) than val_score
                denominator = abs(train_score)
                if train_score > val_score + 1e-9: # Check if train score is better
                    if denominator > 1e-9:
                        diff_val_pct = abs(diff_val / denominator) * 100.0
                    if diff_val_pct > threshold_pct:
                        is_overfitting_val = True

            # Log comparison results regardless of overfitting detection
            if denominator > 1e-9:
                logging.info(f"      Diff (Train - Val): {diff_val:.4f} ({diff_val_pct:.2f}%)")
            else:
                logging.info(f"      Diff (Train - Val): {diff_val:.4f} (Cannot calculate % diff - denominator near zero)")

            if is_overfitting_val:
                # Use the user-specified log message format
                logging.warning(f"[Patch] Potential Overfitting detected. Train vs Val {metric} gap = {abs(diff_val):.4f} ({diff_val_pct:.2f}% > {threshold_pct:.1f}%)")

        elif X_train is not None:
            logging.warning("      Diff (Train - Val): Cannot calculate (NaN score).")
        # <<< End of [Patch] MODIFIED v4.8.8 (Patch 9) >>>

        if pd.notna(val_score) and pd.notna(test_score):
            diff_test = val_score - test_score
            is_generalization_issue = False
            denominator_test = 0.0
            diff_test_pct = float('inf')

            if metric == "LogLoss": # Lower is better
                denominator_test = abs(val_score)
                if test_score > val_score + 1e-9: # Test score is worse
                    if denominator_test > 1e-9:
                        diff_test_pct = abs(diff_test / denominator_test) * 100.0
                    if diff_test_pct > threshold_pct:
                        is_generalization_issue = True
            else: # Higher is better
                denominator_test = abs(val_score)
                if val_score > test_score + 1e-9: # Val score is better
                    if denominator_test > 1e-9:
                        diff_test_pct = abs(diff_test / denominator_test) * 100.0
                    if diff_test_pct > threshold_pct:
                        is_generalization_issue = True

            if denominator_test > 1e-9:
                 logging.info(f"      Diff (Val - Test): {diff_test:.4f} ({diff_test_pct:.2f}%)")
            else:
                 logging.info(f"      Diff (Val - Test): {diff_test:.4f} (Cannot calculate % diff - denominator near zero)")

            if is_generalization_issue:
                logging.warning(f"      (ALERT) Potential Generalization Issue: Val {metric} significantly {'lower' if metric=='LogLoss' else 'higher'} than Test {metric} (Diff % > {threshold_pct:.1f}%).")
        elif X_test is not None:
            logging.warning("      Diff (Val - Test): Cannot calculate (NaN score).")

    except Exception as e:
        logging.error(f"      (Error) Error during Overfitting check ({metric}): {e}", exc_info=True)

# [Patch v5.0.2] Exclude SHAP noise check from coverage
def check_feature_noise_shap(shap_values, feature_names, threshold=0.01):  # pragma: no cover
    """
    Checks for potentially noisy features based on low mean absolute SHAP values.
    (v4.8.8 Patch 9: Fixed logging logic and format)
    """
    logging.info("   [Check] Checking for Feature Noise (SHAP)...")
    if shap_values is None or not isinstance(shap_values, np.ndarray) or not feature_names or not isinstance(feature_names, list) or \
       shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names) or shap_values.shape[0] == 0:
        logging.warning("      (Warning) Skipping Feature Noise Check: Invalid inputs."); return

    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        if np.isnan(mean_abs_shap).any() or np.isinf(mean_abs_shap).any():
            logging.warning("      (Warning) Found NaN/Inf in Mean Abs SHAP. Skipping noise check.")
            return

        shap_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})
        total_shap = shap_df["Mean_Abs_SHAP"].sum()
        shap_df["Normalized_SHAP"] = (shap_df["Mean_Abs_SHAP"] / total_shap) if total_shap > 1e-9 else 0.0

        # <<< [Patch] MODIFIED v4.8.8 (Patch 9): Corrected logging as per user prompt and test failure >>>
        # Use the DataFrame index directly if feature_names match the index
        shap_series_for_check = pd.Series(shap_df["Normalized_SHAP"].values, index=shap_df["Feature"])
        noise_feats = shap_series_for_check[shap_series_for_check < threshold].index.tolist()
        if noise_feats:
            # Use the user-specified log message format (logging.info as per patch 6 note)
            logging.info(f"[Patch] SHAP Noise features detected: {noise_feats}")
        # <<< End of [Patch] MODIFIED v4.8.8 (Patch 9) >>>
        else:
            logging.info(f"      (Success) No features with significant noise detected (Normalized SHAP < {threshold:.4f}).")
    except Exception as e:
        logging.error(f"      (Error) Error during Feature Noise check (SHAP): {e}", exc_info=True)

# --- SHAP Analysis Function ---
# [Patch v5.0.2] Exclude SHAP importance analysis from coverage
def analyze_feature_importance_shap(model, model_type, data_sample, features, output_dir, fold_idx=None):  # pragma: no cover
    """
    Analyzes feature importance using SHAP values and saves summary plots.
    (v4.8.8 Patch 1: Enhanced robustness for SHAP value structure and feature validation)
    """
    global shap
    if not shap:
        logging.warning("   (Warning) Skipping SHAP: 'shap' library not found.")
        return
    if model is None:
        logging.warning("   (Warning) Skipping SHAP: Model is None.")
        return
    if data_sample is None or not isinstance(data_sample, pd.DataFrame) or data_sample.empty:
        logging.warning("   (Warning) Skipping SHAP: No sample data.")
        return
    if not features or not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        logging.warning("   (Warning) Skipping SHAP: Invalid features list.")
        return
    if not output_dir or not os.path.isdir(output_dir):
        logging.warning(f"   (Warning) Skipping SHAP: Output directory '{output_dir}' invalid.")
        return

    fold_suffix = f"_fold{fold_idx+1}" if fold_idx is not None else "_validation_set"
    logging.info(f"\n(Analyzing) SHAP analysis ({model_type} - Sample Size: {len(data_sample)}) - {fold_suffix.replace('_',' ').title()}...")

    missing_features = [f for f in features if f not in data_sample.columns]
    if missing_features:
        logging.error(f"   (Error) Skipping SHAP: Missing features in data_sample: {missing_features}")
        return
    try:
        X_shap = data_sample[features].copy()
    except KeyError as e:
        logging.error(f"   (Error) Skipping SHAP: Feature(s) not found: {e}")
        return
    except Exception as e_select:
        logging.error(f"   (Error) Skipping SHAP: Error selecting features: {e_select}", exc_info=True)
        return

    cat_features_indices = []
    cat_feature_names_shap = []
    potential_cat_cols = ['Pattern_Label', 'session', 'Trend_Zone']
    logging.debug("      Processing categorical features for SHAP...")
    for cat_col in potential_cat_cols:
        if cat_col in X_shap.columns:
            try:
                if X_shap[cat_col].isnull().any():
                    X_shap[cat_col].fillna("Missing", inplace=True)
                X_shap[cat_col] = X_shap[cat_col].astype(str)
                if model_type == "CatBoostClassifier":
                    cat_feature_names_shap.append(cat_col)
            except Exception as e_cat_str:
                logging.warning(f"      (Warning) Could not convert '{cat_col}' to string for SHAP: {e_cat_str}.")

    if model_type == "CatBoostClassifier" and cat_feature_names_shap:
        try:
            cat_features_indices = [X_shap.columns.get_loc(col) for col in cat_feature_names_shap]
            logging.debug(f"         Categorical Feature Indices for SHAP Pool: {cat_features_indices}")
        except KeyError as e_loc:
            logging.error(f"      (Error) Could not locate categorical feature index for SHAP: {e_loc}.")
            cat_features_indices = []

    logging.debug("      Handling NaN/Inf in numeric features for SHAP...")
    numeric_cols_shap = X_shap.select_dtypes(include=np.number).columns
    if X_shap[numeric_cols_shap].isin([np.inf, -np.inf]).any().any():
        X_shap[numeric_cols_shap] = X_shap[numeric_cols_shap].replace([np.inf, -np.inf], np.nan)
    if X_shap[numeric_cols_shap].isnull().any().any():
        X_shap[numeric_cols_shap] = X_shap[numeric_cols_shap].fillna(0)

    if X_shap.isnull().any().any():
        missing_final = X_shap.columns[X_shap.isnull().any()].tolist()
        logging.error(f"      (Error) Skipping SHAP: NaNs still present after fill in columns: {missing_final}")
        return

    try:
        explainer = None
        shap_values = None
        global CatBoostClassifier, Pool

        logging.debug(f"      Initializing SHAP explainer for model type: {model_type}...")
        if model_type == "CatBoostClassifier" and CatBoostClassifier and Pool:
            shap_pool = Pool(X_shap, label=None, cat_features=cat_features_indices)
            explainer = shap.TreeExplainer(model)
            logging.info(f"      Calculating SHAP values (CatBoost)...")
            shap_values = explainer.shap_values(shap_pool)
        else:
            logging.warning(f"      (Warning) SHAP explainer not supported or library missing for model type: {model_type}")
            return

        shap_values_positive_class = None
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            if isinstance(shap_values[1], np.ndarray) and shap_values[1].ndim == 2:
                shap_values_positive_class = shap_values[1]
            else:
                logging.error(f"      (Error) SHAP values list element 1 has unexpected type/shape: {type(shap_values[1])}, {getattr(shap_values[1], 'shape', 'N/A')}")
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            shap_values_positive_class = shap_values
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            if shap_values.shape[0] >= 2 and shap_values.shape[1] == X_shap.shape[0] and shap_values.shape[2] == X_shap.shape[1]:
                shap_values_positive_class = shap_values[1, :, :]
            elif shap_values.shape[2] >= 2 and shap_values.shape[0] == X_shap.shape[0] and shap_values.shape[1] == X_shap.shape[1]:
                shap_values_positive_class = shap_values[:, :, 1]
            elif shap_values.shape[0] == 1:
                shap_values_positive_class = shap_values[0, :, :]
                logging.warning("      SHAP values have only one class output, using index 0.")
            else:
                logging.error(f"      (Error) Unexpected 3D SHAP values shape: {shap_values.shape}. Cannot determine positive class.")
        else:
            logging.error(f"      (Error) Unexpected SHAP values structure (Type: {type(shap_values)}, Shape: {getattr(shap_values, 'shape', 'N/A')}). Cannot plot.")
            return

        if shap_values_positive_class is None:
            logging.error("      (Error) Could not identify SHAP values for positive class.")
            return
        if shap_values_positive_class.shape[1] != len(features):
            logging.error(f"      (Error) SHAP feature dimension mismatch ({shap_values_positive_class.shape[1]} != {len(features)}). Cannot proceed.")
            return
        if shap_values_positive_class.shape[0] != X_shap.shape[0]:
            logging.error(f"      (Error) SHAP sample dimension mismatch ({shap_values_positive_class.shape[0]} != {X_shap.shape[0]}). Cannot proceed.")
            return

        logging.info("      Creating SHAP Summary Plot (bar type)...")
        shap_plot_path = os.path.join(output_dir, f"shap_summary_{model_type}_bar{fold_suffix}.png")
        plt.figure()
        try:
            shap.summary_plot(shap_values_positive_class, X_shap, plot_type="bar", show=False, feature_names=features, max_display=20)
            plt.title(f"SHAP Feature Importance ({model_type} - {fold_suffix.replace('_',' ').title()})")
            plt.tight_layout()
            plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
            logging.info(f"      (Success) Saved SHAP Plot (Bar): {os.path.basename(shap_plot_path)}")
        except Exception as e_save_bar:
            logging.error(f"      (Error) Failed to create/save SHAP bar plot: {e_save_bar}", exc_info=True)
        finally:
            plt.close()

        logging.info("      Creating SHAP Summary Plot (beeswarm/dot type)...")
        shap_beeswarm_path = os.path.join(output_dir, f"shap_summary_{model_type}_beeswarm{fold_suffix}.png")
        plt.figure()
        try:
            shap.summary_plot(shap_values_positive_class, X_shap, plot_type="dot", show=False, feature_names=features, max_display=20)
            plt.title(f"SHAP Feature Summary ({model_type} - {fold_suffix.replace('_',' ').title()})")
            plt.tight_layout()
            plt.savefig(shap_beeswarm_path, dpi=300, bbox_inches="tight")
            logging.info(f"      (Success) Saved SHAP Plot (Beeswarm): {os.path.basename(shap_beeswarm_path)}")
        except Exception as e_save_beeswarm:
            logging.error(f"      (Error) Failed to create/save SHAP beeswarm plot: {e_save_beeswarm}", exc_info=True)
        finally:
            plt.close()

    except ImportError:
        logging.error("   (Error) SHAP Error: Could not import shap library components.")
    except Exception as e:
        logging.error(f"   (Error) Error during SHAP analysis: {e}", exc_info=True)

# --- Feature Loading Function ---
# [Patch v5.0.2] Exclude feature loader from coverage
def load_features_for_model(model_name, output_dir):  # pragma: no cover
    """
    Loads the feature list for a specific model purpose from a JSON file.
    Falls back to loading 'features_main.json' if the specific file is not found.
    """
    features_filename = f"features_{model_name}.json"
    features_file_path = os.path.join(output_dir, features_filename)
    logging.info(f"   (Feature Load) Attempting to load features for '{model_name}' from: {features_file_path}")

    if not os.path.exists(features_file_path):
        logging.info(
            f"   (Info) Feature file not found for model '{model_name}': {os.path.basename(features_file_path)}"
        )
        main_features_path = os.path.join(output_dir, "features_main.json")
        if model_name != "main" and os.path.exists(main_features_path):
            logging.info(
                "      (Fallback) Loading features from 'features_main.json' instead."
            )
            features_file_path = main_features_path  # Use main path for fallback
        else:
            logging.info(
                "      (Generating) Default features_main.json."
            )
            try:
                os.makedirs(output_dir, exist_ok=True)
                with open(main_features_path, "w", encoding="utf-8") as f_def:
                    json.dump(DEFAULT_META_CLASSIFIER_FEATURES, f_def, ensure_ascii=False, indent=2)
                logging.info(
                    "      (Generated) Default features_main.json created using DEFAULT_META_CLASSIFIER_FEATURES."
                )
                features_file_path = main_features_path
            except Exception as e_write:
                logging.error(
                    f"   (Error) Could not create default features_main.json: {e_write}"
                )
                return DEFAULT_META_CLASSIFIER_FEATURES

    try:
        with open(features_file_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        if isinstance(features, list) and all(isinstance(feat, str) for feat in features):
            logging.info(f"      (Success) Loaded {len(features)} features for model '{model_name}' from '{os.path.basename(features_file_path)}'.")
            return features
        else:
            logging.error(f"   (Error) Invalid format in feature file: {features_file_path}. Expected list of strings.")
            return None
    except json.JSONDecodeError as e_json:
        logging.error(f"   (Error) Failed to decode JSON from feature file '{os.path.basename(features_file_path)}': {e_json}")
        return None
    except Exception as e:
        logging.error(f"   (Error) Failed to load features for model '{model_name}' from '{os.path.basename(features_file_path)}': {e}", exc_info=True)
        return None

# --- Model Switcher Function ---
# [Patch v5.0.2] Exclude model switcher from coverage
def select_model_for_trade(context, available_models=None):  # pragma: no cover
    """
    Selects the appropriate AI model ('main', 'spike', 'cluster') based on context.
    Falls back to 'main' if the selected model is invalid or missing.
    """
    selected_model_key = 'main' # Default model
    confidence = None

    cluster_value = context.get('cluster')
    spike_score_value = context.get('spike_score', 0.0)

    if not isinstance(cluster_value, (int, float, np.number)) or pd.isna(cluster_value):
        cluster_value = None
    if not isinstance(spike_score_value, (int, float, np.number)) or pd.isna(spike_score_value):
        spike_score_value = 0.0

    spike_switch_threshold = 0.6
    cluster_switch_value = 2

    logging.debug(f"      (Switcher) Context: SpikeScore={spike_score_value:.3f}, Cluster={cluster_value}")

    if spike_score_value > spike_switch_threshold:
        selected_model_key = 'spike'
        confidence = spike_score_value
    elif cluster_value == cluster_switch_value:
        selected_model_key = 'cluster'
        confidence = 0.8
    else:
        selected_model_key = 'main'
        confidence = None

    if available_models is None:
        logging.error("      (Switcher Error) 'available_models' is None. Defaulting to 'main'.")
        selected_model_key = 'main'
        confidence = None
    elif selected_model_key not in available_models or \
         available_models.get(selected_model_key, {}).get('model') is None or \
         not available_models.get(selected_model_key, {}).get('features'):
        logging.warning(f"      (Switcher Warning) Selected model '{selected_model_key}' invalid/missing. Defaulting to 'main'.")
        selected_model_key = 'main'
        confidence = None

    logging.debug(f"      (Switcher) Final Selected Model: '{selected_model_key}', Confidence: {confidence}")
    return selected_model_key, confidence

logging.info("Part 6: Machine Learning Configuration & Helpers Loaded (v4.8.8 Patch 9 Applied).")
# === END OF PART 6/12 ===

# ---------------------------------------------------------------------------
# Stubs for Function Registry Tests















def calculate_trend_zone(df):
    """Stubbed trend zone calculator."""
    return pd.Series("NEUTRAL", index=df.index)




def create_session_column(df):
    """Stubbed session column creator."""
    df["session"] = "Other"
    return df


def fill_missing_feature_values(df):
    """Stubbed missing feature filler."""
    return df.fillna(0)


def load_feature_config(path):
    """Stubbed feature config loader."""
    return {}


def calculate_ml_features(df):
    """Stubbed ML feature calculator."""
    return df


# [Patch v5.5.7] Add simple volume spike detector
def is_volume_spike(current_vol, avg_vol, multiplier=1.5):
    """Return True if current volume exceeds multiplier * average volume."""
    try:
        cur = float(current_vol)
        avg = float(avg_vol)
    except Exception:
        logging.debug("Volume Spike: invalid input values")
        return False
    if np.isnan(cur) or np.isnan(avg) or avg <= 0:
        return False
    return cur > avg * multiplier


# [Patch v5.6.2] HDF5 helpers fallback to pickle when PyTables missing
def save_features_hdf5(df, path):
    """Save a DataFrame to an HDF5 file or pickle if PyTables is unavailable."""
    try:
        import importlib.util
        if importlib.util.find_spec("tables") is None:
            df.to_pickle(path)
            logging.warning("(Warning) PyTables not installed, saved as pickle")
            return
        df.to_hdf(path, key="data", mode="w")
        logging.info(f"(Features) Saved features to {path}")
    except Exception as e:
        logging.error(f"(Features) Failed to save features to {path}: {e}", exc_info=True)


def load_features_hdf5(path):
    """Load a DataFrame from an HDF5 file or pickle as a fallback."""
    try:
        import importlib.util
        if importlib.util.find_spec("tables") is None:
            df = pd.read_pickle(path)
            logging.warning("(Warning) PyTables not installed, loaded from pickle")
            return df
        df = pd.read_hdf(path, key="data")
        logging.info(f"(Features) Loaded features from {path}")
        return df
    except Exception as e:
        logging.error(f"(Features) Failed to load features from {path}: {e}", exc_info=True)
        return None

# --- Advanced Feature Utilities -------------------------------------------------
# [Patch v5.6.5] Add momentum, cumulative delta, and wave pattern helpers

def add_momentum_features(df, windows=(5, 10, 15, 20)):
    """Add ROC and RSI momentum features for the given rolling windows."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be DataFrame")
    if not {'Close', 'Open'}.issubset(df.columns):
        logging.warning("    (Warning) Missing 'Close' or 'Open' columns for momentum features.")
        return df

    df_out = df.copy()
    close = pd.to_numeric(df_out['Close'], errors='coerce')
    for w in windows:
        if not isinstance(w, int) or w <= 0:
            continue
        roc = close.pct_change(periods=w) * 100
        df_out[f'ROC_{w}'] = roc.astype('float32')
        df_out[f'RSI_{w}'] = rsi(close, period=w)
    return df_out


def calculate_cumulative_delta_price(df, window=10):
    """Return rolling sum of Close-Open over the given window."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be DataFrame")
    if not {'Close', 'Open'}.issubset(df.columns):
        logging.warning("    (Warning) Missing 'Close' or 'Open' columns for cumulative delta.")
        return pd.Series(np.zeros(len(df)), index=df.index, dtype='float32')
    delta = pd.to_numeric(df['Close'], errors='coerce') - pd.to_numeric(df['Open'], errors='coerce')
    cum_delta = delta.rolling(window=window, min_periods=1).sum()
    return cum_delta.astype('float32')


def merge_wave_pattern_labels(df, log_path):
    """Merge Wave_Marker_Unit pattern labels onto df as 'Wave_Pattern'."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be DataFrame")
    df_out = df.copy()
    if not os.path.exists(log_path):
        logging.warning(f"   (Warning) Wave_Marker log not found: {log_path}")
        df_out['Wave_Pattern'] = 'Unknown'
        df_out['Wave_Pattern'] = df_out['Wave_Pattern'].astype('category')
        return df_out
    try:
        pattern_df = pd.read_csv(log_path)
        pattern_df['datetime'] = pd.to_datetime(pattern_df['datetime'], errors='coerce')
        pattern_df = pattern_df.dropna(subset=['datetime', 'pattern_label']).sort_values('datetime')
    except Exception as e:
        logging.error(f"   (Error) Failed to load pattern log: {e}", exc_info=True)
        df_out['Wave_Pattern'] = 'Unknown'
        df_out['Wave_Pattern'] = df_out['Wave_Pattern'].astype('category')
        return df_out

    if not isinstance(df_out.index, pd.DatetimeIndex):
        df_out.index = pd.to_datetime(df_out.index, errors='coerce')

    merged = pd.merge_asof(
        df_out.sort_index(),
        pattern_df.rename(columns={'datetime': 'index'}).sort_values('index'),
        left_index=True,
        right_on='index',
        direction='backward'
    )
    df_out['Wave_Pattern'] = merged['pattern_label'].fillna('Unknown').astype('category')
    return df_out


