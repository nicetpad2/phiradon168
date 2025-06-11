import pandas as pd
import pytest

import backtest_engine as be


def test_run_backtest_engine_success(monkeypatch):
    """ควรรีเทิร์น DataFrame เมื่อ simulation ทำงานสำเร็จ"""
    price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    trade_df = pd.DataFrame({'pnl': [1.0]})

    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: price_df)
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))

    result = be.run_backtest_engine(pd.DataFrame())
    assert result.equals(trade_df)


def test_run_backtest_engine_fail_load(monkeypatch):
    """หากโหลดราคาล้มเหลวต้องยก RuntimeError"""
    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError('no')))
    with pytest.raises(RuntimeError):
        be.run_backtest_engine(pd.DataFrame())


def test_run_backtest_engine_empty_log(monkeypatch):
    """เมื่อ trade log ว่างต้องยก RuntimeError"""
    price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: price_df)
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, pd.DataFrame()))
    with pytest.raises(RuntimeError):
        be.run_backtest_engine(pd.DataFrame())


def test_run_backtest_engine_index_conversion(monkeypatch):
    """ควรแปลง index เป็น DatetimeIndex เพื่อให้มีคุณสมบัติ .tz"""
    price_df = pd.DataFrame({
        'Open': [1],
        'High': [1],
        'Low': [1],
        'Close': [1]
    }, index=['2024-01-01 00:00:00'])
    trade_df = pd.DataFrame({'pnl': [1.0]})

    captured = {}

    def fake_simulation(df, **k):
        captured['index_is_dt'] = isinstance(df.index, pd.DatetimeIndex)
        captured['tz_attr'] = getattr(df.index, 'tz', None)
        return None, trade_df

    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: price_df)
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', fake_simulation)

    result = be.run_backtest_engine(pd.DataFrame())
    assert result.equals(trade_df)
    assert captured['index_is_dt']
    assert 'tz_attr' in captured
