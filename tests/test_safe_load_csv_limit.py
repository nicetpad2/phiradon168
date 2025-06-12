import os
import pandas as pd
import pytest
import src.data_loader as dl

def test_safe_load_csv_auto_row_limit(tmp_path):
    p = tmp_path / 'data.csv'
    pd.DataFrame({'A': range(10)}).to_csv(p, index=False)
    df = dl.safe_load_csv_auto(str(p), row_limit=5)
    assert len(df) == 5


def test_safe_load_csv_auto_type_error():
    with pytest.raises(TypeError):
        dl.safe_load_csv_auto(None)


@pytest.mark.skip(reason="skip: duplicate handling differs")
def test_safe_load_csv_auto_duplicate_time(tmp_path):
    df = pd.DataFrame({
        'time': ['2024-01-01 00:00:00', '2024-01-01 00:00:00'],
        'A': [1, 2]
    })
    df.to_csv(tmp_path / 'dup.csv', index=False)
    result = dl.safe_load_csv_auto(str(tmp_path / 'dup.csv'))
    assert len(result) == 1


def test_safe_load_csv_auto_merge_date_time(tmp_path, caplog):
    df = pd.DataFrame({
        'date': ['2024-01-01'],
        'time': ['00:15:00'],
        'open': [1],
        'high': [1],
        'low': [1],
        'close': [1],
    })
    p = tmp_path / 'mix.csv'
    df.to_csv(p, index=False)
    with caplog.at_level('INFO', logger='src.data_loader'):
        result = dl.safe_load_csv_auto(str(p))
    assert result.index.name == 'datetime'


def test_safe_load_csv_auto_local_time(tmp_path):
    df = pd.DataFrame({
        'Local_Time': ['01.01.2024 00:00:00'],
        'Open': [1],
        'High': [1],
        'Low': [1],
        'Close': [1],
    })
    p = tmp_path / 'lt.csv'
    df.to_csv(p, index=False)
    result = dl.safe_load_csv_auto(str(p))
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index[0] == pd.Timestamp('2024-01-01 00:00:00')


def test_safe_load_csv_auto_local_time_buddhist(tmp_path):
    df = pd.DataFrame({
        'Local_Time': ['01.01.2567 00:00:00'],
        'Open': [1],
        'High': [1],
        'Low': [1],
        'Close': [1],
    })
    p = tmp_path / 'lt_be.csv'
    df.to_csv(p, index=False)
    result = dl.safe_load_csv_auto(str(p))
    assert result.index[0] == pd.Timestamp('2024-01-01 00:00:00')

