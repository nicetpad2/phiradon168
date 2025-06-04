import os
import pandas as pd
from src.data_loader import load_data_cached


def test_load_data_cached_parquet(tmp_path):
    csv_path = tmp_path / 'sample.csv'
    df = pd.DataFrame({
        'Date': ['2024-01-01'],
        'Timestamp': ['2024-01-01 00:00:00'],
        'Open': [1.0],
        'High': [1.0],
        'Low': [1.0],
        'Close': [1.0],
    })
    df.to_csv(csv_path, index=False)

    df_loaded = load_data_cached(str(csv_path), 'M1', cache_format='parquet')
    cache_file = csv_path.with_suffix('.parquet')
    # บางสภาพแวดล้อมอาจไม่รองรับ pyarrow หรือ fastparquet
    if cache_file.exists():
        os.remove(csv_path)
        df_cached = load_data_cached(str(csv_path), 'M1', cache_format='parquet')
        assert not df_cached.empty
    else:
        assert not df_loaded.empty
