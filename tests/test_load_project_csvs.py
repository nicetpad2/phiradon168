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
