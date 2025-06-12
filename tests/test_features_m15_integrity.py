import logging
import pandas as pd
from src import features

def test_m15_trend_zone_buddhist_year_warns(caplog):
    df = pd.DataFrame({
        'Date': ['25670101', '25670101'],
        'Timestamp': ['00:00:00', '00:15:00'],
        'Close': [1.0, 1.1]
    })
    with caplog.at_level(logging.WARNING):
        result = features.calculate_m15_trend_zone(df)
    assert (result['Trend_Zone'] == 'NEUTRAL').all()
    assert 'RSI calculation skipped' in ' '.join(caplog.messages)

