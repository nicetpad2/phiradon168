import pandas as pd
import pytest
import importlib
import src.data_loader as data_loader


def test_load_data(monkeypatch):
    # Create temporary CSV files
    df = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
    tmp_m1 = 'tmp_m1.csv'
    tmp_m15 = 'tmp_m15.csv'
    df.to_csv(tmp_m1, index=False)
    df.to_csv(tmp_m15, index=False)

    monkeypatch.setattr(data_loader, 'CSV_PATH_M1', tmp_m1, raising=False)
    monkeypatch.setattr(data_loader, 'CSV_PATH_M15', tmp_m15, raising=False)

    df1, df2 = data_loader.load_data()
    assert df1.equals(df)
    assert df2.equals(df)
