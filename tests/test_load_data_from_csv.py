import pandas as pd
import pytest
from src.data_loader import load_data_from_csv




def test_load_data_from_csv_timestamp(tmp_path):
    df = pd.DataFrame({
        'Timestamp': ['2024-01-02 00:00:00'],
        'Open': [1],
        'High': [1],
        'Low': [1],
        'Close': [1],
        'Volume': [1],
    })
    p = tmp_path / 'ts.csv'
    df.to_csv(p, index=False)
    res = load_data_from_csv(str(p))
    assert res.index[0] == pd.Timestamp('2024-01-02 00:00:00')
    assert isinstance(res.index, pd.DatetimeIndex)


def test_load_data_from_csv_missing_cols(tmp_path):
    df = pd.DataFrame({'Time': ['2024-01-01']})
    p = tmp_path / 'bad.csv'
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_data_from_csv(str(p))
