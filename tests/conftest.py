import pytest
import pandas as pd

@pytest.fixture
def simple_m1_df():
    data = {
        'Open': [10, 10.1, 10.2, 10.3, 10.4],
        'High': [10.2, 10.3, 10.4, 10.5, 10.6],
        'Low': [9.8, 9.9, 10.0, 10.1, 10.2],
        'Close': [10.1, 10.2, 10.3, 10.4, 10.5],
        'Volume': [100, 110, 120, 130, 140],
        'Datetime': pd.date_range('2023-01-01', periods=5, freq='T')
    }
    df = pd.DataFrame(data)
    df.set_index('Datetime', inplace=True)
    return df
