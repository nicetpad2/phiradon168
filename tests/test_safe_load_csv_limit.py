import os
import pandas as pd
import pytest
import src.data_loader as dl

def test_safe_load_csv_auto_row_limit(tmp_path):
    p = tmp_path / 'data.csv'
    pd.DataFrame({'A': range(10)}).to_csv(p, index=False)
    df = dl.safe_load_csv_auto(str(p), row_limit=5)
    assert len(df) == 5

def test_safe_load_csv_auto_env_limit(tmp_path, monkeypatch):
    p = tmp_path / 'data.csv'
    pd.DataFrame({'A': range(10)}).to_csv(p, index=False)
    monkeypatch.setenv('DATA_ROW_LIMIT', '3')
    df = dl.safe_load_csv_auto(str(p))
    assert len(df) == 3
    monkeypatch.delenv('DATA_ROW_LIMIT', raising=False)


def test_safe_load_csv_auto_type_error():
    with pytest.raises(TypeError):
        dl.safe_load_csv_auto(None)

