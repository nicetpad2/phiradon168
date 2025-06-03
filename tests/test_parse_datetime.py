import os
import sys
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.data_loader import parse_datetime_safely


def test_parse_datetime_multiple_formats():
    series = pd.Series([
        '20240101 12:30:00',
        '01/02/2024 07:00:00',
    ])
    result = parse_datetime_safely(series)
    assert result.dtype == 'datetime64[ns]'
    assert result.iloc[0] == pd.Timestamp('2024-01-01 12:30:00')
    assert result.iloc[1] == pd.Timestamp('2024-02-01 07:00:00')


def test_parse_datetime_invalid_type():
    with pytest.raises(TypeError):
        parse_datetime_safely(['2024-01-01'])
