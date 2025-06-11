import pandas as pd
import pytest
import logging

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
    """เมื่อ trade log ว่างควรรีเทิร์น DataFrame ว่าง"""
    price_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: price_df)
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, pd.DataFrame()))
    result = be.run_backtest_engine(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert result.empty


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


def test_run_backtest_engine_generates_trend_and_signals(monkeypatch):
    m1_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    m15_df = pd.DataFrame({'Close': [1]}, index=[pd.Timestamp('2024-01-01')])
    trade_df = pd.DataFrame({'pnl': [1.0]})

    def fake_read_csv(path, *a, **k):
        return m1_df if path == be.DATA_FILE_PATH_M1 else m15_df

    monkeypatch.setattr(be.pd, 'read_csv', fake_read_csv)
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)

    flags = {}

    def fake_trend(df):
        flags['trend_called'] = True
        return pd.DataFrame({'Trend_Zone': ['UP']}, index=df.index)

    def fake_entry(df, cfg):
        flags['has_trend'] = 'Trend_Zone' in df.columns
        return df.assign(
            Entry_Long=0,
            Entry_Short=0,
            Trade_Tag='t',
            Signal_Score=0.1,
            Trade_Reason='r'
        )

    def fake_sim(df, **k):
        flags['cols'] = df.columns.tolist()
        return None, trade_df

    monkeypatch.setattr(be, 'calculate_m15_trend_zone', fake_trend)
    monkeypatch.setattr(be, 'calculate_m1_entry_signals', fake_entry)
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', fake_sim)

    result = be.run_backtest_engine(pd.DataFrame())
    assert result.equals(trade_df)
    assert flags.get('trend_called')
    assert flags.get('has_trend')
    for col in ['Trend_Zone', 'Entry_Long', 'Entry_Short', 'Trade_Tag', 'Signal_Score', 'Trade_Reason']:
        assert col in flags.get('cols', [])


def test_run_backtest_engine_drops_duplicate_trend_index(monkeypatch, caplog):
    m1_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    m15_df = pd.DataFrame({'Close': [1, 2]}, index=[pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01')])
    trade_df = pd.DataFrame({'pnl': [1.0]})

    def fake_read_csv(path, *a, **k):
        return m1_df if path == be.DATA_FILE_PATH_M1 else m15_df

    monkeypatch.setattr(be.pd, 'read_csv', fake_read_csv)
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
    monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long=0, Entry_Short=0, Trade_Tag='t', Signal_Score=0.0, Trade_Reason='r'))
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))

    def fake_trend(df):
        return pd.DataFrame({'Trend_Zone': ['UP', 'DOWN']}, index=df.index)

    monkeypatch.setattr(be, 'calculate_m15_trend_zone', fake_trend)

    with caplog.at_level(logging.INFO):
        result = be.run_backtest_engine(pd.DataFrame())

    assert result.equals(trade_df)
    assert any('duplicate index rows' in msg for msg in caplog.messages)


def test_run_backtest_engine_sorts_trend_index(monkeypatch, caplog):
    m1_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]})
    m15_df = pd.DataFrame({'Close': [1, 2]}, index=[pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01')])
    trade_df = pd.DataFrame({'pnl': [1.0]})

    def fake_read_csv(path, *a, **k):
        return m1_df if path == be.DATA_FILE_PATH_M1 else m15_df

    monkeypatch.setattr(be.pd, 'read_csv', fake_read_csv)
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
    monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long=0, Entry_Short=0, Trade_Tag='t', Signal_Score=0.0, Trade_Reason='r'))
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))

    def fake_trend(df):
        return pd.DataFrame({'Trend_Zone': ['UP', 'DOWN']}, index=df.index)

    monkeypatch.setattr(be, 'calculate_m15_trend_zone', fake_trend)

    with caplog.at_level(logging.INFO):
        result = be.run_backtest_engine(pd.DataFrame())

    assert result.equals(trade_df)
    assert any('Sorted Trend Zone DataFrame index' in msg for msg in caplog.messages)


def test_run_backtest_engine_sorts_m1_index(monkeypatch, caplog):
    m1_df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [1], 'Close': [1]},
                         index=[pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-01')])
    trade_df = pd.DataFrame({'pnl': [1.0]})

    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: m1_df)
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
    monkeypatch.setattr(be, 'calculate_m15_trend_zone', lambda df: pd.DataFrame({'Trend_Zone': ['UP']}, index=df.index))
    monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long=0, Entry_Short=0, Trade_Tag='t', Signal_Score=0.0, Trade_Reason='r'))
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))

    with caplog.at_level(logging.INFO):
        result = be.run_backtest_engine(pd.DataFrame())

    assert result.equals(trade_df)
    assert any('index M1 ไม่เรียงลำดับเวลา' in msg for msg in caplog.messages)


def test_run_backtest_engine_drops_duplicate_m1_index(monkeypatch, caplog):
    idx = [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01')]
    m1_df = pd.DataFrame({'Open': [1, 2], 'High': [1, 2], 'Low': [1, 2], 'Close': [1, 2]}, index=idx)
    trade_df = pd.DataFrame({'pnl': [1.0]})

    monkeypatch.setattr(be.pd, 'read_csv', lambda *a, **k: m1_df)
    monkeypatch.setattr(be, 'engineer_m1_features', lambda df, **k: df)
    monkeypatch.setattr(be, 'calculate_m15_trend_zone', lambda df: pd.DataFrame({'Trend_Zone': ['UP', 'UP']}, index=df.index))
    monkeypatch.setattr(be, 'calculate_m1_entry_signals', lambda df, cfg: df.assign(Entry_Long=0, Entry_Short=0, Trade_Tag='t', Signal_Score=0.0, Trade_Reason='r'))
    monkeypatch.setattr(be, 'run_backtest_simulation_v34', lambda df, **k: (None, trade_df))

    with caplog.at_level(logging.INFO):
        result = be.run_backtest_engine(pd.DataFrame())

    assert result.equals(trade_df)
    assert any('index ซ้ำซ้อนในข้อมูลราคา M1' in msg for msg in caplog.messages)
