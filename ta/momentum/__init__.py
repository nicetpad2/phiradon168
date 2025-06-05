import pandas as pd

class RSIIndicator:
    def __init__(self, close, window=14, fillna=False):
        self._close = pd.Series(close, dtype='float32')
        self._window = window
        self._fillna = fillna

    def rsi(self):
        diff = self._close.diff()
        gain = diff.clip(lower=0)
        loss = -diff.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/self._window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self._window, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        if self._fillna:
            rsi = rsi.fillna(0)
        return rsi.astype('float32')
