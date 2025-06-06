# Vendored from python-ta version 0.11.0 on 2025-06-06
# Upstream: https://github.com/bukosabino/ta
import pandas as pd
import numpy as np

class MACD:
    def __init__(self, close, window_slow=26, window_fast=12, window_sign=9, fillna=False):
        self._close = pd.Series(close, dtype='float32')
        self._slow = window_slow
        self._fast = window_fast
        self._sign = window_sign
        self._fillna = fillna

    def _ema(self, series, span):
        return series.ewm(span=span, adjust=False).mean()

    def macd(self):
        ema_fast = self._ema(self._close, self._fast)
        ema_slow = self._ema(self._close, self._slow)
        return (ema_fast - ema_slow).astype('float32')

    def macd_signal(self):
        return self.macd().ewm(span=self._sign, adjust=False).mean().astype('float32')

    def macd_diff(self):
        line = self.macd()
        signal = self.macd_signal()
        return (line - signal).astype('float32')


class ADXIndicator:
    """Calculate Average Directional Index (ADX)."""

    def __init__(self, high, low, close, window=14, fillna=False):
        self._high = pd.Series(high, dtype='float32')
        self._low = pd.Series(low, dtype='float32')
        self._close = pd.Series(close, dtype='float32')
        self._window = window
        self._fillna = fillna

    def _true_range(self):
        high = self._high
        low = self._low
        prev_close = self._close.shift()
        hl = high - low
        hc = (high - prev_close).abs()
        lc = (low - prev_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        if not tr.empty:
            tr.iloc[0] = hl.iloc[0]
        return tr

    def _dm_plus(self):
        up = self._high.diff()
        down = -self._low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        return pd.Series(plus_dm, index=self._high.index)

    def _dm_minus(self):
        up = self._high.diff()
        down = -self._low.diff()
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        return pd.Series(minus_dm, index=self._high.index)

    def adx(self):
        tr = self._true_range()
        atr = tr.ewm(alpha=1 / self._window, adjust=False, min_periods=self._window).mean()
        plus_dm = self._dm_plus().ewm(alpha=1 / self._window, adjust=False, min_periods=self._window).mean()
        minus_dm = self._dm_minus().ewm(alpha=1 / self._window, adjust=False, min_periods=self._window).mean()

        plus_di = 100 * plus_dm / atr
        minus_di = 100 * minus_dm / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / self._window, adjust=False, min_periods=self._window).mean()

        if self._fillna:
            adx = adx.fillna(0)
        return adx.astype('float32')
