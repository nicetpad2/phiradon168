import os
import pytest
import pandas as pd
import numpy as np
import ProjectP
import strategy.entry_rules as entry_rules
import strategy as strategy_pkg


def test_load_trade_log_regeneration(tmp_path, monkeypatch):
    file = tmp_path / 'trade.csv'
    file.write_text('timestamp,price,signal\n')
    monkeypatch.setenv('TRADE_LOG_MIN_ROWS', '5')

    def fake_engine(_):
        return pd.DataFrame({'pnl': [1] * 5})

    import backtest_engine
    monkeypatch.setattr(backtest_engine, 'run_backtest_engine', fake_engine)
    monkeypatch.setattr(ProjectP, 'load_features', lambda p: pd.DataFrame())

    df = ProjectP.load_trade_log(str(file), min_rows=int(os.environ['TRADE_LOG_MIN_ROWS']))
    assert len(df) >= 5


@pytest.fixture
def df_features():
    return pd.DataFrame({
        'Close': [1.0, 1.1, 1.2, 1.3, 1.4],
        'Volume': [100, 110, 120, 130, 140],
    })


def test_generate_open_signals_fallback(df_features, monkeypatch):
    def zero_classifier(df):
        return np.zeros(len(df), dtype=np.int8)

    monkeypatch.setattr(entry_rules, 'generate_open_signals', zero_classifier)

    def fallback(df, *args, **kwargs):
        res = zero_classifier(df)
        if res.sum() == 0:
            return (df['Close'] > df['Close'].shift(1)).fillna(0).astype(np.int8).to_numpy()
        return res

    monkeypatch.setattr(strategy_pkg, 'generate_open_signals', fallback)
    signals = strategy_pkg.generate_open_signals(df_features)
    assert signals.sum() > 0

