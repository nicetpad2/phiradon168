import pandas as pd
from src import features


def test_tag_engulfing_patterns_basic():
    df = pd.DataFrame({'Open': [1, 2, 1.5], 'Close': [2, 1, 2.5]})
    res = features.tag_engulfing_patterns(df)
    assert 'Engulfing' in res.columns
    assert res['Engulfing'].dtype.name == 'category'

