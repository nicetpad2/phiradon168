import pandas as pd
from strategy import trend_filter


def test_apply_trend_filter():
    df = pd.DataFrame({
        'Trend_Zone': ['UP', 'DOWN', 'NEUTRAL'],
        'Entry_Long': [1, 1, 1],
        'Entry_Short': [1, 1, 1],
    })
    result = trend_filter.apply_trend_filter(df)
    assert result['Entry_Long'].tolist() == [1, 0, 0]
    assert result['Entry_Short'].tolist() == [0, 1, 0]
