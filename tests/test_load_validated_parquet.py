import pandas as pd
import src.main as main


def test_load_validated_csv_parquet(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'Date': ['20240101'],
        'Timestamp': ['00:00:00'],
        'Open': [1.0],
        'High': [1.0],
        'Low': [1.0],
        'Close': [1.0]
    })
    path = tmp_path / 'sample.parquet'
    # Stub parquet functions to avoid pyarrow dependency
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', lambda self, p: self.to_csv(str(p).replace('.parquet', '.csv')))
    monkeypatch.setattr(pd, 'read_parquet', lambda p: pd.read_csv(str(p).replace('.parquet', '.csv')))

    df.to_parquet(path)
    loaded = main.load_validated_csv(str(path), 'M1')
    for col in df.columns:
        assert col in loaded.columns
        pd.testing.assert_series_equal(loaded[col].astype(str), df[col].astype(str), check_dtype=False)
