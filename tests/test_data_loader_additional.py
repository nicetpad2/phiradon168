import os
import json
import pandas as pd
import numpy as np
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.data_loader as dl

def test_load_app_config_missing(tmp_path):
    conf_path = tmp_path / 'nonexistent.json'
    result = dl.load_app_config(str(conf_path))
    assert result == {}


def test_load_app_config_valid(tmp_path):
    conf_data = {"x": 1}
    conf_file = tmp_path / 'cfg.json'
    conf_file.write_text(json.dumps(conf_data), encoding='utf-8')
    result = dl.load_app_config(str(conf_file))
    assert result == conf_data


def test_safe_set_datetime_basic():
    df = pd.DataFrame(index=[0])
    dl.safe_set_datetime(df, 0, 'Date', '2023-01-02')
    assert df['Date'].dtype == 'datetime64[ns]'
    assert df.loc[0, 'Date'] == pd.Timestamp('2023-01-02')


def test_safe_set_datetime_handles_timezone():
    df = pd.DataFrame({'Date': [pd.Timestamp('2024-01-01', tz='UTC')]})
    dl.safe_set_datetime(df, 0, 'Date', pd.Timestamp('2024-01-02', tz='UTC'))
    assert df['Date'].dtype == 'datetime64[ns]'
    assert df.loc[0, 'Date'] == pd.Timestamp('2024-01-02')


def test_safe_set_datetime_naive_tz():
    df = pd.DataFrame(index=[0])
    dl.safe_set_datetime(df, 0, 'Date', '2024-01-02 07:00:00', naive_tz='Asia/Bangkok')
    assert df.loc[0, 'Date'] == pd.Timestamp('2024-01-02 00:00:00')


def test_rsi_returns_nan_when_ta_missing(monkeypatch):
    series = pd.Series([1, 2, 3], dtype='float32')
    monkeypatch.setattr(dl, 'ta', None, raising=False)
    import importlib
    features = importlib.import_module('src.features')
    monkeypatch.setattr(features, 'ta', None, raising=False)
    result = features.rsi(series, period=5)
    assert result.isna().all()

def test_simple_converter_edge_cases():
    assert dl.simple_converter(np.inf) == "Infinity"
    assert dl.simple_converter(-np.inf) == "-Infinity"
    assert dl.simple_converter(pd.NaT) is None
    class Dummy:
        def __str__(self):
            return "dummy"
    assert dl.simple_converter(Dummy()) == "dummy"


def test_simple_converter_timedelta():
    td = pd.Timedelta("1 days")
    assert dl.simple_converter(td) == "1 days 00:00:00"


def test_safe_get_global_returns_default():
    assert dl.safe_get_global('NON_EXISTENT_VAR', 123) == 123

def test_setup_output_directory_creates(tmp_path):
    base = tmp_path
    result = dl.setup_output_directory(str(base), 'out')
    assert os.path.isdir(result)


def test_validate_m1_data_path_ok(tmp_path):
    p = tmp_path / 'XAUUSD_M1.csv'
    p.write_text('a,b\n1,2', encoding='utf-8')
    assert dl.validate_m1_data_path(str(p)) is True


def test_validate_m1_data_path_prep(tmp_path):
    p = tmp_path / 'final_data_m1_v32_walkforward_prep_data_NORMAL.csv.gz'
    pd.DataFrame({'Open':[1], 'High':[1], 'Low':[1], 'Close':[1]}).to_csv(p, compression='gzip')
    assert dl.validate_m1_data_path(str(p)) is True


def test_validate_m1_data_path_wrong_name(tmp_path, caplog):
    p = tmp_path / 'bad.csv'
    p.write_text('a,b\n1,2', encoding='utf-8')
    with caplog.at_level('ERROR'):
        assert not dl.validate_m1_data_path(str(p))
    assert 'Unexpected M1 data file' in caplog.text


def test_load_raw_data_m1_bad_path(tmp_path, caplog):
    p = tmp_path / 'wrong.csv'
    pd.DataFrame({'A': [1]}).to_csv(p)
    with caplog.at_level('ERROR'):
        res = dl.load_raw_data_m1(str(p))
    assert res is None
    assert 'Unexpected M1 data file' in caplog.text


def test_validate_m15_data_path_ok(tmp_path):
    p = tmp_path / 'XAUUSD_M15.csv'
    p.write_text('a,b\n1,2', encoding='utf-8')
    assert dl.validate_m15_data_path(str(p)) is True


def test_validate_m15_data_path_wrong_name(tmp_path, caplog):
    p = tmp_path / 'bad.csv'
    p.write_text('a,b\n1,2', encoding='utf-8')
    with caplog.at_level('ERROR'):
        assert not dl.validate_m15_data_path(str(p))
    assert 'Unexpected M15 data file' in caplog.text


def test_load_raw_data_m15_bad_path(tmp_path, caplog):
    p = tmp_path / 'wrong.csv'
    pd.DataFrame({'A': [1]}).to_csv(p)
    with caplog.at_level('ERROR'):
        res = dl.load_raw_data_m15(str(p))
    assert res is None
    assert 'Unexpected M15 data file' in caplog.text


def test_load_final_m1_data_valid(tmp_path):
    df = pd.DataFrame(
        {
            'Open': [1.0],
            'High': [1.1],
            'Low': [0.9],
            'Close': [1.0],
            'ATR_14': [0.1],
        },
        index=pd.date_range('2023-01-01', periods=1, freq='min', tz='UTC')
    )
    path = tmp_path / 'final_data_m1_v32_walkforward.csv.gz'
    df.to_csv(path, compression='gzip')
    trade_log = pd.DataFrame({'datetime': [pd.Timestamp('2023-01-01', tz='UTC')]})
    loaded = dl.load_final_m1_data(str(path), trade_log)
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.tz == trade_log['datetime'].dt.tz


def test_load_final_m1_data_corrupt(tmp_path, caplog):
    path = tmp_path / 'final_data_m1_v32_walkforward.csv.gz'
    path.write_text('corrupt')
    trade_log = pd.DataFrame({'datetime': [pd.Timestamp('2023-01-01', tz='UTC')]})
    with caplog.at_level('ERROR'):
        res = dl.load_final_m1_data(str(path), trade_log)
    assert res is None
    assert 'Failed to load' in caplog.text or 'Error' in caplog.text


def test_load_final_m1_data_missing_cols(tmp_path):
    df = pd.DataFrame({'Open': [1]})
    path = tmp_path / 'final_data_m1_v32_walkforward.csv.gz'
    df.to_csv(path, compression='gzip')
    trade_log = pd.DataFrame({'datetime': [pd.Timestamp('2023-01-01', tz='UTC')]})
    assert dl.load_final_m1_data(str(path), trade_log) is None


def test_load_data_max_rows(tmp_path, caplog):
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=10, freq='min').date,
        'Timestamp': pd.date_range('2023-01-01', periods=10, freq='min'),
        'Open': np.arange(10, dtype=float),
        'High': np.arange(10, dtype=float) + 0.1,
        'Low': np.arange(10, dtype=float) - 0.1,
        'Close': np.arange(10, dtype=float)
    })
    csv_path = tmp_path / 'data.csv'
    df.to_csv(csv_path, index=False)
    dtypes = {'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32'}
    with caplog.at_level('INFO'):
        loaded = dl.load_data(str(csv_path), 'M1', dtypes=dtypes, max_rows=5)
    assert loaded.shape[0] == 5
    assert 'max_rows=5' in caplog.text


def test_load_data_max_rows_none(tmp_path):
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=8, freq='min').date,
        'Timestamp': pd.date_range('2023-01-01', periods=8, freq='min'),
        'Open': np.arange(8, dtype=float),
        'High': np.arange(8, dtype=float) + 0.1,
        'Low': np.arange(8, dtype=float) - 0.1,
        'Close': np.arange(8, dtype=float)
    })
    csv_path = tmp_path / 'data.csv'
    df.to_csv(csv_path, index=False)
    dtypes = {'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32'}
    loaded = dl.load_data(str(csv_path), 'M1', dtypes=dtypes)
    assert loaded.shape[0] == 8


def test_load_data_volume_dtype(tmp_path):
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=5, freq='min').date,
        'Timestamp': pd.date_range('2023-01-01', periods=5, freq='min'),
        'Open': np.arange(5, dtype=float),
        'High': np.arange(5, dtype=float) + 0.1,
        'Low': np.arange(5, dtype=float) - 0.1,
        'Close': np.arange(5, dtype=float),
        'Volume': np.linspace(1.0, 5.0, num=5),
    })
    csv_path = tmp_path / 'data.csv'
    df.to_csv(csv_path, index=False)
    loaded = dl.load_data(str(csv_path), 'M1')
    assert loaded['Volume'].dtype == 'float32'


def test_load_data_missing_file(tmp_path, caplog):
    missing = tmp_path / 'missing.csv'
    with caplog.at_level('CRITICAL'):
        with pytest.raises(SystemExit):
            dl.load_data(str(missing), 'M1')
    assert 'ไม่พบไฟล์' in caplog.text
