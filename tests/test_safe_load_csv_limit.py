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


def test_safe_load_csv_auto_duplicate_index(tmp_path):
    df = pd.DataFrame({'A': [1, 2]}, index=[0, 0])
    df.to_csv(tmp_path / 'dup.csv')
    with pytest.raises(dl.DataValidationError):
        dl.safe_load_csv_auto(str(tmp_path / 'dup.csv'))

