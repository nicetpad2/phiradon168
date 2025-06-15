import pandas as pd
from strategy.entry_rules import generate_open_signals


def test_generate_open_signals_short():
    df = pd.DataFrame({
        "Close": [1.0, 0.9, 0.8],
        "MACD_hist": [-0.1, -0.1, -0.1],
        "RSI": [40, 45, 45],
        "MA_fast": [1.0, 0.9, 0.8],
        "MA_slow": [1.1, 1.1, 1.1],
    })
    signals = generate_open_signals(df, use_macd=False, use_rsi=False, allow_short=True)
    assert any(signals == -1)
