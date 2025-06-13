import pandas as pd
import src.data_loader as dl


def test_safe_load_csv_auto_root_lookup(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    df = dl.safe_load_csv_auto('XAUUSD_M1.csv', row_limit=20)
    assert isinstance(df, pd.DataFrame)
    assert df is not None
