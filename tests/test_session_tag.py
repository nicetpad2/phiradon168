import os
import sys
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.features as features


def test_get_session_tag_basic():
    ts = pd.Timestamp('2024-01-01 07:00', tz='UTC')
    assert features.get_session_tag(ts) == 'Asia/London'


def test_engineer_session_vectorized_distribution():
    idx = pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC')
    df = pd.DataFrame({'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0}, index=idx)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        result = features.engineer_m1_features(df)
    categories = set(result['session'].unique())
    expected = {'Asia', 'Asia/London', 'London', 'London/NY', 'NY'}
    assert expected.issubset(categories)
    assert result['session'].dtype.name == 'category'


def test_engineer_session_with_rangeindex():
    df = pd.DataFrame({'Open': 1.0, 'High': 1.0, 'Low': 1.0, 'Close': 1.0}, index=range(5))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        result = features.engineer_m1_features(df)
    assert 'session' in result.columns
    assert result['session'].dtype.name == 'category'
