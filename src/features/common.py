import logging
import pandas as pd
import numpy as np

try:  # [Patch v5.8.2] Handle missing ta library gracefully
    import ta
    _TA_AVAILABLE = True
except ImportError:  # pragma: no cover - environment may not have ta installed
    class _DummySubmodule:
        pass

    class _DummyTA:
        def __init__(self):
            self.volatility = _DummySubmodule()
            self.trend = _DummySubmodule()
            self.momentum = _DummySubmodule()

    ta = _DummyTA()
    _TA_AVAILABLE = False
    logging.warning("'ta' library not found. Technical indicators will return NaN.")

from sklearn.cluster import KMeans  # For context column calculation
from sklearn.preprocessing import StandardScaler  # For context column calculation
import gc  # For memory management
from src.utils.gc_utils import maybe_collect
from functools import lru_cache
from src.utils.sessions import get_session_tag, get_session_tags_vectorized  # [Patch v5.1.3]
from src.utils import get_env_float, load_json_with_comments

_rsi_cache = {}  # [Patch v4.8.12] Cache RSIIndicator per period
_atr_cache = {}  # [Patch v4.8.12] Cache AverageTrueRange per period
_sma_cache = {}  # [Patch v4.8.12] Cache SMA results
_m15_trend_cache = {}


def reset_indicator_caches() -> None:
    """Clear cached indicator objects before each fold."""
    _rsi_cache.clear()
    _atr_cache.clear()
    _sma_cache.clear()
    _m15_trend_cache.clear()

# Ensure global configurations are accessible if run independently
DEFAULT_ROLLING_Z_WINDOW_M1 = 300
DEFAULT_ATR_ROLLING_AVG_PERIOD = 50
DEFAULT_PATTERN_BREAKOUT_Z_THRESH = 2.0
DEFAULT_PATTERN_REVERSAL_BODY_RATIO = 0.5
DEFAULT_PATTERN_STRONG_TREND_Z_THRESH = 1.0
DEFAULT_PATTERN_CHOPPY_CANDLE_RATIO = 0.3
DEFAULT_PATTERN_CHOPPY_WICK_RATIO = 0.6
DEFAULT_M15_TREND_EMA_FAST = 50
DEFAULT_M15_TREND_EMA_SLOW = 200
DEFAULT_M15_TREND_RSI_PERIOD = 14
DEFAULT_M15_TREND_RSI_UP = 51
DEFAULT_M15_TREND_RSI_DOWN = 49  # [Patch v5.6.4]
DEFAULT_TIMEFRAME_MINUTES_M1 = 1
DEFAULT_MIN_SIGNAL_SCORE_ENTRY = 1.0  # [Patch v5.3.9]
DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO = 1.8
DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO = 0.75
DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R = 0.5
DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R = 1.0
DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R = 0.3
DEFAULT_ADAPTIVE_TSL_START_ATR_MULT = 1.5

ROLLING_Z_WINDOW_M1 = DEFAULT_ROLLING_Z_WINDOW_M1
ATR_ROLLING_AVG_PERIOD = DEFAULT_ATR_ROLLING_AVG_PERIOD
PATTERN_BREAKOUT_Z_THRESH = DEFAULT_PATTERN_BREAKOUT_Z_THRESH
PATTERN_REVERSAL_BODY_RATIO = DEFAULT_PATTERN_REVERSAL_BODY_RATIO
PATTERN_STRONG_TREND_Z_THRESH = DEFAULT_PATTERN_STRONG_TREND_Z_THRESH
PATTERN_CHOPPY_CANDLE_RATIO = DEFAULT_PATTERN_CHOPPY_CANDLE_RATIO
PATTERN_CHOPPY_WICK_RATIO = DEFAULT_PATTERN_CHOPPY_WICK_RATIO
M15_TREND_EMA_FAST = DEFAULT_M15_TREND_EMA_FAST
M15_TREND_EMA_SLOW = DEFAULT_M15_TREND_EMA_SLOW
M15_TREND_RSI_PERIOD = DEFAULT_M15_TREND_RSI_PERIOD
M15_TREND_RSI_UP = DEFAULT_M15_TREND_RSI_UP
M15_TREND_RSI_DOWN = DEFAULT_M15_TREND_RSI_DOWN
TIMEFRAME_MINUTES_M1 = DEFAULT_TIMEFRAME_MINUTES_M1
MIN_SIGNAL_SCORE_ENTRY = DEFAULT_MIN_SIGNAL_SCORE_ENTRY
ADAPTIVE_TSL_HIGH_VOL_RATIO = DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO
ADAPTIVE_TSL_LOW_VOL_RATIO = DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO
ADAPTIVE_TSL_DEFAULT_STEP_R = DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R
ADAPTIVE_TSL_HIGH_VOL_STEP_R = DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R
ADAPTIVE_TSL_LOW_VOL_STEP_R = DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R
ADAPTIVE_TSL_START_ATR_MULT = DEFAULT_ADAPTIVE_TSL_START_ATR_MULT

META_CLASSIFIER_FEATURES = []
SESSION_TIMES_UTC = {"Asia": (22, 8), "London": (7, 16), "NY": (13, 21)}
__all__ = [
    'logging', 'pd', 'np', 'ta', '_TA_AVAILABLE',
    'KMeans', 'StandardScaler', 'maybe_collect', 'lru_cache',
    'get_session_tag', 'get_session_tags_vectorized',
    '_rsi_cache', '_atr_cache', '_sma_cache', '_m15_trend_cache',
    'reset_indicator_caches',
    'ROLLING_Z_WINDOW_M1', 'ATR_ROLLING_AVG_PERIOD',
    'PATTERN_BREAKOUT_Z_THRESH', 'PATTERN_REVERSAL_BODY_RATIO',
    'PATTERN_STRONG_TREND_Z_THRESH', 'PATTERN_CHOPPY_CANDLE_RATIO',
    'get_env_float', 'load_json_with_comments',
    'PATTERN_CHOPPY_WICK_RATIO', 'M15_TREND_EMA_FAST', 'M15_TREND_EMA_SLOW',
    'M15_TREND_RSI_PERIOD', 'M15_TREND_RSI_UP', 'M15_TREND_RSI_DOWN',
    'TIMEFRAME_MINUTES_M1', 'MIN_SIGNAL_SCORE_ENTRY',
    'ADAPTIVE_TSL_HIGH_VOL_RATIO', 'ADAPTIVE_TSL_LOW_VOL_RATIO',
    'ADAPTIVE_TSL_DEFAULT_STEP_R', 'ADAPTIVE_TSL_HIGH_VOL_STEP_R',
    'ADAPTIVE_TSL_LOW_VOL_STEP_R', 'ADAPTIVE_TSL_START_ATR_MULT',
    'META_CLASSIFIER_FEATURES', 'SESSION_TIMES_UTC'
]
