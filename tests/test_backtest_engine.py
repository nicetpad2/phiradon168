import pandas as pd
import pytest

import backtest_engine as be


def test_run_backtest_engine_success(monkeypatch):
    """ควรรีเทิร์น DataFrame เมื่อ simulation ทำงานสำเร็จ"""
    price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    trade_df = pd.DataFrame({'pnl': [1.0]})

    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: price_df)
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))

    result = be.run_backtest_engine(pd.DataFrame())
    assert result.equals(trade_df)


def test_run_backtest_engine_fail_load(monkeypatch):
    """หากโหลดราคาล้มเหลวต้องยก RuntimeError"""
    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError('no')))
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: (_ for _ in ()).throw(AssertionError('should not be called')))
    with pytest.raises(RuntimeError):
        be.run_backtest_engine(pd.DataFrame())


def test_run_backtest_engine_empty_log(monkeypatch):
    """เมื่อ trade log ว่างต้องยก RuntimeError"""
    price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: price_df)
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
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
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', fake_simulation)

    result = be.run_backtest_engine(pd.DataFrame())
    assert result.equals(trade_df)
    assert captured['index_is_dt']
    assert 'tz_attr' in captured


def test_run_backtest_engine_passes_fold_params(monkeypatch):
    """ควรส่ง fold_config และ current_fold_index ให้ simulation"""
    price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    trade_df = pd.DataFrame({'pnl': [1.0]})

    captured = {}

    def fake_simulation(df, **kwargs):
        captured['fold_config'] = kwargs.get('fold_config')
        captured['current_fold_index'] = kwargs.get('current_fold_index')
        return None, trade_df

    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: price_df)
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', fake_simulation)

    result = be.run_backtest_engine(pd.DataFrame())
    assert result.equals(trade_df)
    assert captured['fold_config'] == be.DEFAULT_FOLD_CONFIG
    assert captured['current_fold_index'] == be.DEFAULT_FOLD_INDEX


def test_run_backtest_engine_calls_feature_engineering(monkeypatch):
    """ควรเรียก engineer_m1_features ก่อน simulation"""
    price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    trade_df = pd.DataFrame({'pnl': [1.0]})

    calls = {'count': 0}

    def fake_engineer(df, **k):
        calls['count'] += 1
        return df

    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: price_df)
    monkeypatch.setattr(be, 'engineer_m1_features', fake_engineer)
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))

    result = be.run_backtest_engine(pd.DataFrame())
    assert result.equals(trade_df)
    assert calls['count'] == 1
