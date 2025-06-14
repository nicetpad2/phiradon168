import pandas as pd
import pytest
from src.data_loader import load_project_csvs

def test_load_project_csvs():
    m1, m15 = load_project_csvs(row_limit=5)
    assert isinstance(m1, pd.DataFrame)
    assert isinstance(m15, pd.DataFrame)
    assert m1 is not None
    assert m15 is not None
    for col in ["Open", "High", "Low", "Close"]:
        assert col.lower() in m1.columns or col in m1.columns
        assert col.lower() in m15.columns or col in m15.columns


def test_load_project_csvs_clean(monkeypatch):
    calls = []

    def fake_validate(path, output=None, required_cols=None):
        calls.append(path)
        return pd.DataFrame({
            'Time': pd.date_range('2020-01-01', periods=3),
            'Open': [1, 1, 1],
            'High': [1, 1, 1],
            'Low': [1, 1, 1],
            'Close': [1, 1, 1],
            'Volume': [1, 1, 1]
        })

    import src.csv_validator as csv_validator
    monkeypatch.setattr(csv_validator, 'validate_and_convert_csv', fake_validate)
    m1, m15 = load_project_csvs(row_limit=2, clean=True)
    assert len(calls) == 2
    assert len(m1) == 2
    assert len(m15) == 2
