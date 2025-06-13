import pandas as pd
from src import entry_rules, config
from src.constants import ColumnName


def test_dynamic_threshold_env(monkeypatch):
    df = pd.DataFrame({'signal_score': [0.4, 0.7], ColumnName.CLOSE: [1, 1]})
    monkeypatch.setenv('MIN_SIGNAL_SCORE_ENTRY', '0.5')
    signals = entry_rules.generate_open_signals(df)
    assert signals.tolist() == [0, 1]
    monkeypatch.delenv('MIN_SIGNAL_SCORE_ENTRY', raising=False)


def test_ma_fallback(monkeypatch):
    df = pd.DataFrame({'signal_score': [0, 0, 0, 0], ColumnName.CLOSE: [1, 2, 3, 4]})
    monkeypatch.setenv('MIN_SIGNAL_SCORE_ENTRY', '5.0')
    monkeypatch.setattr(config, 'FAST_MA_PERIOD', 2)
    monkeypatch.setattr(config, 'SLOW_MA_PERIOD', 3)
    signals = entry_rules.generate_open_signals(df)
    assert signals.sum() > 0
    monkeypatch.delenv('MIN_SIGNAL_SCORE_ENTRY', raising=False)
