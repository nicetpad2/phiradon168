# Vendored from python-ta version 0.11.0 on 2025-06-06
# Upstream: https://github.com/bukosabino/ta
import pandas as pd

class AverageTrueRange:
    def __init__(self, high, low, close, window=14, fillna=False):
        self._high = pd.Series(high, dtype='float32')
        self._low = pd.Series(low, dtype='float32')
        self._close = pd.Series(close, dtype='float32')
        self._window = window
        self._fillna = fillna

    def average_true_range(self):
        high = self._high
        low = self._low
        close = self._close
        hl = high - low
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        if not tr.empty:
            tr.iloc[0] = hl.iloc[0]
        atr = tr.ewm(alpha=1/self._window, adjust=False, min_periods=self._window).mean()
        if self._fillna:
            atr = atr.fillna(0)
        return atr.astype('float32')
