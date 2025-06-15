from .common import *
import os
from .engineering import engineer_m1_features
from .technical import *

def calculate_trend_zone(df):
    """Stubbed trend zone calculator."""
    return pd.Series("NEUTRAL", index=df.index)




def create_session_column(df):
    """Create ``session`` column using :func:`get_session_tag`.

    This replaces the previous stub implementation that filled all rows
    with ``"Other"``. The function now attempts to convert the DataFrame
    index to ``DatetimeIndex`` (if necessary) and maps each timestamp to
    a trading session tag. If tagging fails, the session is set to
    ``"Other"``.
    """

    # [Patch v6.8.13] Proper session tagging implementation
    if df is None:
        logging.warning("create_session_column received None, returning empty DataFrame")
        return pd.DataFrame({"session": pd.Categorical([], categories=["Asia", "London", "NY", "Other", "N/A"])} )
    if df.empty:
        df["session"] = pd.Categorical([], categories=["Asia", "London", "NY", "Other", "N/A"])
        return df

    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce", format="mixed")
        df["session"] = get_session_tags_vectorized(df.index).astype(
            "category"
        ).cat.set_categories(["Asia", "London", "NY", "Other", "N/A"], ordered=False)
    except Exception as e:
        logging.error(f"   (Error) Session calculation failed: {e}", exc_info=True)
        df["session"] = pd.Categorical(["Other"] * len(df), categories=["Asia", "London", "NY", "Other", "N/A"])
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

# [Patch vX.Y.Z] Parquet helpers for faster feature loading
def save_features_parquet(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a Parquet file."""
    try:
        df.to_parquet(path)
        logging.info(f"(Features) Saved features to {path}")
    except Exception as e:
        logging.error(f"(Features) Failed to save features to {path}: {e}", exc_info=True)


def load_features_parquet(path: str) -> pd.DataFrame | None:
    """Load a DataFrame from a Parquet file."""
    try:
        df = pd.read_parquet(path)
        logging.info(f"(Features) Loaded features from {path}")
        return df
    except Exception as e:
        logging.error(f"(Features) Failed to load features from {path}: {e}", exc_info=True)
        return None

# [Patch v6.8.5] Generic helpers choosing format
def save_features(df: pd.DataFrame, path: str, fmt: str = "parquet") -> None:
    """Save DataFrame in the specified format."""
    fmt_lower = (fmt or "parquet").lower()
    if fmt_lower == "hdf5":
        save_features_hdf5(df, path)
    elif fmt_lower == "parquet":
        save_features_parquet(df, path)
    else:
        df.to_csv(path, index=False)
        logging.info(f"(Features) Saved features to {path} as CSV")


def load_features(path: str, fmt: str = "parquet") -> pd.DataFrame | None:
    """Load DataFrame from the specified format."""
    fmt_lower = (fmt or "parquet").lower()
    if fmt_lower == "hdf5":
        return load_features_hdf5(path)
    if fmt_lower == "parquet":
        return load_features_parquet(path)
    try:
        df = pd.read_csv(path)
        logging.info(f"(Features) Loaded features from {path} as CSV")
        return df
    except Exception as e:
        logging.error(f"(Features) Failed to load features from {path}: {e}", exc_info=True)
        return None


# [Patch v6.9.7] Add helper to load or engineer M1 features with caching
def load_or_engineer_m1_features(
    df_m1: pd.DataFrame,
    cache_path: str | None = None,
    fmt: str = "parquet",
) -> pd.DataFrame:
    """Load cached M1 features if available, otherwise engineer and cache them."""
    if cache_path and os.path.exists(cache_path):
        cached = load_features(cache_path, fmt=fmt)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            logging.info(f"(Cache) Loaded M1 features from {cache_path}")
            return cached

    features_df = engineer_m1_features(df_m1)

    if cache_path:
        try:
            save_features(features_df, cache_path, fmt=fmt)
            logging.info(f"(Cache) Saved engineered M1 features to {cache_path}")
        except Exception as e:
            logging.error(
                f"(Cache) Failed to save features to {cache_path}: {e}",
                exc_info=True,
            )

    return features_df

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


# [Patch v6.1.7] Engulfing candlestick pattern tagging
def tag_engulfing_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Label bullish/bearish engulfing patterns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be DataFrame")
    df_out = df.copy()
    if not {"Open", "Close"}.issubset(df_out.columns):
        logging.warning("(Warning) Missing Open/Close for engulfing pattern")
        df_out["Engulfing"] = "None"
        df_out["Engulfing"] = df_out["Engulfing"].astype("category")
        return df_out

    open_p = pd.to_numeric(df_out["Open"], errors="coerce")
    close_p = pd.to_numeric(df_out["Close"], errors="coerce")
    prev_open = open_p.shift(1)
    prev_close = close_p.shift(1)
    bullish = (
        (prev_close < prev_open)
        & (close_p > open_p)
        & (close_p >= prev_open)
        & (open_p <= prev_close)
    )
    bearish = (
        (prev_close > prev_open)
        & (close_p < open_p)
        & (close_p <= prev_open)
        & (open_p >= prev_close)
    )
    df_out["Engulfing"] = np.select(
        [bullish, bearish], ["Bullish", "Bearish"], default="None"
    )
    df_out["Engulfing"] = df_out["Engulfing"].astype("category")
    return df_out


# [Patch v6.4.3] Build feature catalog from sample M1 data
def build_feature_catalog(data_dir: str, output_dir: str) -> list:
    """Generate numeric feature list from the raw M1 CSV."""
    m1_path = os.path.join(data_dir, "XAUUSD_M1.csv")
    if not os.path.exists(m1_path):
        raise FileNotFoundError(f"M1 data not found: {m1_path}")
    # [Patch v6.9.29] ใช้ข้อมูลทุกแถวจาก CSV
    df_sample = pd.read_csv(m1_path)
    features = [
        c
        for c in df_sample.columns
        if c not in {"datetime", "is_tp", "is_sl", "Date", "Timestamp"}
        and pd.api.types.is_numeric_dtype(df_sample[c])
    ]  # [Patch v6.7.3] skip Date/Timestamp columns
    return features


