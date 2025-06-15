from .common import *
import sys

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
    if not isinstance(series, pd.Series):
        logging.error(f"RSI Error: Input must be a pandas Series, got {type(series)}")
        raise TypeError("Input must be a pandas Series.")
    # [Patch v4.8.12] Use module-level cache for RSIIndicator
    # [Patch v5.8.1] Ensure insufficient data warning is logged even when 'ta' is missing
    if series.empty:
        logging.debug("RSI: Input series is empty, returning NaN-aligned series.")
        return pd.Series(np.nan, index=series.index, dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    pkg = sys.modules.get("src.features")
    ta_lib = getattr(pkg, "ta", ta)
    ta_available = getattr(pkg, "_TA_AVAILABLE", _TA_AVAILABLE) and ta_lib is not None
    if not ta_available:
        logging.warning("   (Warning) Using pandas fallback RSI because 'ta' library not loaded.")
        if series_numeric.empty or len(series_numeric) < period:
            logging.warning(
                f"   (Warning) RSI calculation skipped: Not enough valid data points ({len(series_numeric)} < {period})."
            )
            return pd.Series(np.nan, index=series.index, dtype='float32')
        delta = series_numeric.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss_safe = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss_safe
        rsi_series = (100 - 100 / (1 + rs)).fillna(100)
        return rsi_series.reindex(series.index).ffill().astype('float32')
    ta_loaded = ta_lib is not None
    if not ta_loaded:
        logging.error("   (Error) RSI calculation failed: 'ta' library not loaded.")
    if series_numeric.empty or len(series_numeric) < period:
        logging.warning(
            f"   (Warning) RSI calculation skipped: Not enough valid data points ({len(series_numeric)} < {period})."
        )
        return pd.Series(np.nan, index=series.index, dtype='float32')
    if not ta_loaded:
        return pd.Series(np.nan, index=series.index, dtype='float32')
    # [Patch v5.5.16] Consolidate duplicate timestamps using last occurrence
    if series_numeric.index.duplicated().any():
        series_numeric = series_numeric.groupby(series_numeric.index).last()
    cache_key = period
    use_ta = hasattr(ta_lib, 'momentum') and hasattr(ta_lib.momentum, 'RSIIndicator')
    if use_ta:
        try:
            if cache_key not in _rsi_cache:
                _rsi_cache[cache_key] = ta_lib.momentum.RSIIndicator(close=series_numeric, window=period, fillna=False)
            else:
                _rsi_cache[cache_key]._close = series_numeric
            rsi_series = _rsi_cache[cache_key].rsi()
        except Exception as e:
            logging.error(f"   (Error) RSI calculation error for period {period}: {e}.", exc_info=True)
            return pd.Series(np.nan, index=series.index, dtype='float32')
    else:
        _rsi_cache.setdefault(cache_key, object())
        delta = series_numeric.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        rsi_series = 100 - (100 / (1 + (avg_gain / avg_loss.replace(0, np.nan))))
    rsi_final = rsi_series.reindex(series.index).ffill().astype('float32')
    del series_numeric, rsi_series
    maybe_collect()
    return rsi_final

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
    if not isinstance(series, pd.Series):
        logging.error(f"MACD Error: Input must be a pandas Series, got {type(series)}")
        raise TypeError("Input must be a pandas Series.")
    if series.empty:
        n = pd.Series(dtype='float32'); return n, n.copy(), n.copy()
    nan = pd.Series(np.nan, index=series.index, dtype='float32')
    if len(series.dropna()) < window_slow:
        logging.debug(f"MACD: Input series too short after dropna ({len(series.dropna())} < {window_slow}).")
        return nan, nan.copy(), nan.copy()
    s = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    pkg = sys.modules.get("src.features")
    ta_lib = getattr(pkg, "ta", ta)
    ta_available = getattr(pkg, "_TA_AVAILABLE", _TA_AVAILABLE) and ta_lib is not None
    if not ta_available:
        logging.warning("   (Warning) Using pandas fallback MACD because 'ta' library not loaded.")
        if s.empty or len(s) < window_slow:
            logging.warning(
                f"   (Warning) MACD calculation skipped: Not enough valid data points ({len(s)} < {window_slow})."
            )
            return nan, nan.copy(), nan.copy()
        ema_fast = s.ewm(span=window_fast, adjust=False, min_periods=1).mean()
        ema_slow = s.ewm(span=window_slow, adjust=False, min_periods=1).mean()
        line = ema_fast - ema_slow
        signal = line.ewm(span=window_sign, adjust=False, min_periods=1).mean()
        diff = line - signal
    else:
        if s.empty or len(s) < window_slow:
            logging.warning(
                f"   (Warning) MACD calculation skipped: Not enough valid data points ({len(s)} < {window_slow})."
            )
            return nan, nan.copy(), nan.copy()
        try:
            ind = ta_lib.trend.MACD(close=s, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign, fillna=False)
            line = ind.macd()
            signal = ind.macd_signal()
            diff = ind.macd_diff()
        except Exception as e:
            logging.error(f"   (Error) MACD calculation error: {e}.", exc_info=True)
            return nan, nan.copy(), nan.copy()
    macd_line_final = line.reindex(series.index).ffill().astype('float32')
    macd_signal_final = signal.reindex(series.index).ffill().astype('float32')
    macd_diff_final = diff.reindex(series.index).ffill().astype('float32')
    del s, line, signal, diff
    maybe_collect()
    return macd_line_final, macd_signal_final, macd_diff_final

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


def volatility_filter(df: pd.DataFrame, period: int = 14, window: int = 50) -> pd.Series:
    """Return True when current ATR >= rolling mean ATR of recent bars."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    col = f"ATR_{period}"
    if col not in df.columns and {"High", "Low", "Close"}.issubset(df.columns):
        df = df.join(atr(df[["High", "Low", "Close"]], period)[[col]])
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    atr_series = pd.to_numeric(df[col], errors="coerce")
    mean_series = atr_series.rolling(window, min_periods=1).mean()
    return (atr_series >= mean_series).fillna(False)


def median_filter(series: pd.Series, window: int = 3) -> pd.Series:
    """Apply simple rolling median filter."""
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    return series.rolling(window, min_periods=1).median()


def bar_range_filter(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Return True when bar range >= threshold."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not {"High", "Low"}.issubset(df.columns):
        return pd.Series(False, index=df.index, dtype=bool)
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    bar_range = high - low
    return (bar_range >= threshold).fillna(False)


def volume_filter(df: pd.DataFrame, window: int = 20, factor: float = 0.7, column: str = "Volume") -> pd.Series:
    """Return True when volume >= rolling average * factor."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if column not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    vol = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    avg = vol.rolling(window, min_periods=1).mean()
    return (vol >= avg * factor).fillna(False)

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

