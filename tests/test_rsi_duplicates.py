import pandas as pd
from src import features


def test_rsi_handles_duplicate_index():
    idx = [pd.Timestamp('2024-01-01 00:00'), pd.Timestamp('2024-01-01 00:01'), pd.Timestamp('2024-01-01 00:01'), pd.Timestamp('2024-01-01 00:02')]
    series = pd.Series([1, 2, 3, 4], index=idx, dtype='float32')
    result = features.rsi(series, period=2)
    assert len(result) == len(series)
    assert not result.isna().all()
