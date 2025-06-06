# Vendored from python-ta version 0.11.0 on 2025-06-06
# Upstream: https://github.com/bukosabino/ta
import pandas as pd

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
