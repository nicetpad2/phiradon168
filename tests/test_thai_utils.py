import os
import sys
import pandas as pd
import pytest
from src.data_loader import convert_thai_datetime, validate_csv_data
from src.utils.hardware import estimate_resource_plan


def test_convert_thai_datetime_basic():
    series = pd.Series(['2567-01-01 12:00:00'])
    res = convert_thai_datetime(series)
    assert res.iloc[0] == pd.Timestamp('2024-01-01 12:00:00', tz='UTC')


def test_convert_thai_datetime_month_name():
    series = pd.Series(['12-มิ.ย.-2563 00:00'])
    res = convert_thai_datetime(series)
    assert res.iloc[0] == pd.Timestamp('2020-06-12 00:00:00', tz='UTC')


def test_convert_thai_datetime_error():
    series = pd.Series(['invalid'])
    with pytest.raises(ValueError):
        convert_thai_datetime(series)


def test_validate_csv_data_ok():
    df = pd.DataFrame({'A': [1], 'B': [2]})
    assert validate_csv_data(df, ['A', 'B']).equals(df)


def test_validate_csv_data_empty():
    with pytest.raises(ValueError):
        validate_csv_data(pd.DataFrame(), ['A'])


def test_validate_csv_data_type_fail():
    df = pd.DataFrame({'Open': ['x'], 'High': [1], 'Low': [0], 'Close': [1], 'Volume': [1]})
    with pytest.raises(ValueError):
        validate_csv_data(df, ['Open', 'High', 'Low', 'Close', 'Volume'])


def test_validate_csv_data_integrity_fail():
    df = pd.DataFrame({'Open': [1], 'High': [0], 'Low': [1], 'Close': [1], 'Volume': [1]})
    with pytest.raises(ValueError):
        validate_csv_data(df, ['Open', 'High', 'Low', 'Close', 'Volume'])


def test_validate_csv_data_negative_volume():
    df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [0], 'Close': [1], 'Volume': [-1]})
    with pytest.raises(ValueError):
        validate_csv_data(df, ['Open', 'High', 'Low', 'Close', 'Volume'])


def test_validate_csv_data_missing_value():
    df = pd.DataFrame({'Open': [1], 'High': [1], 'Low': [0], 'Close': [None], 'Volume': [1]})
    with pytest.raises(ValueError):
        validate_csv_data(df, ['Open', 'High', 'Low', 'Close', 'Volume'])


def test_estimate_resource_plan_defaults(monkeypatch):
    monkeypatch.delitem(sys.modules, 'psutil', raising=False)
    plan = estimate_resource_plan()
    assert 'n_folds' in plan and 'batch_size' in plan
