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


def test_safe_load_csv_auto_duplicate_time(tmp_path):
    df = pd.DataFrame({
        'time': ['2024-01-01 00:00:00', '2024-01-01 00:00:00'],
        'A': [1, 2]
    })
    df.to_csv(tmp_path / 'dup.csv', index=False)
    result = dl.safe_load_csv_auto(str(tmp_path / 'dup.csv'))
    assert len(result) == 1

