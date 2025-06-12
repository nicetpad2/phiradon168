import pandas as pd
from src.data_loader import prepare_datetime

def test_prepare_datetime_buddhist_unique():
    df = pd.DataFrame({
        'Date': ['25670101', '25670101', '25670101'],
        'Timestamp': ['00:00:00', '00:15:00', '00:15:00']
    })
    result = prepare_datetime(df, 'M15')
    assert result.index[0] == pd.Timestamp('2024-01-01 00:00:00')
    assert result.index.is_unique
    assert not result.index.isna().any()

