import pandas as pd
from src.evaluation import calculate_drift_by_period


def test_calculate_drift_by_period_basic():
    idx_train = pd.date_range('2024-01-01', periods=3, freq='D')
    idx_test = pd.date_range('2024-01-02', periods=3, freq='D')
    df_train = pd.DataFrame({'feat': [1.0, 2.0, 3.0]}, index=idx_train)
    df_test = pd.DataFrame({'feat': [1.1, 1.9, 3.1]}, index=idx_test)
    res = calculate_drift_by_period(df_train, df_test, period='D', threshold=0.0)
    assert not res.empty
    assert 'wasserstein' in res.columns

