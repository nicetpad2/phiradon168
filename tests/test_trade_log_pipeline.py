import os
import sys
import pandas as pd
import types
import pytest

from src.trade_log_pipeline import load_or_generate_trade_log
from src.utils.errors import PipelineError


def test_load_existing_trade_log(tmp_path):
    df = pd.DataFrame({'pnl': [1.0, -1.0]})
    path = tmp_path / 'log.csv'
    df.to_csv(path, index=False)
    loaded = load_or_generate_trade_log(str(path), min_rows=2)
    pd.testing.assert_frame_equal(loaded, df)


def test_regenerate_when_missing(monkeypatch, tmp_path):
    path = tmp_path / 'missing.csv'

    def fake_engine(_):
        return pd.DataFrame({'pnl': [1.0] * 5})

    monkeypatch.setitem(sys.modules, 'backtest_engine', types.SimpleNamespace(run_backtest_engine=fake_engine))

    result = load_or_generate_trade_log(str(path), min_rows=5)
    assert len(result) == 5
    assert path.exists()


def test_regenerate_empty_returns_original(monkeypatch, tmp_path):
    path = tmp_path / 'missing.csv'

    def fake_engine(_):
        return pd.DataFrame()

    monkeypatch.setitem(
        sys.modules,
        'backtest_engine',
        types.SimpleNamespace(run_backtest_engine=fake_engine)
    )

    result = load_or_generate_trade_log(str(path), min_rows=1)
    assert result.empty
