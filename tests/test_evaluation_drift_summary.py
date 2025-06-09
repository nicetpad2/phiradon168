import pandas as pd
from src.evaluation import calculate_drift_summary

def test_calculate_drift_summary_warning(caplog):
    idx = pd.date_range('2024-01-01', periods=3, freq='D')
    train_df = pd.DataFrame({'feat': [1.0, 2.0, 3.0]}, index=idx)
    test_df = pd.DataFrame({'feat': [10.0, 11.0, 12.0]}, index=idx)
    with caplog.at_level('WARNING', logger='src.evaluation'):
        res = calculate_drift_summary(train_df, test_df, threshold=0.0)
    assert {'D', 'W'} <= set(res['period_type'])
    assert res['drift'].any()
    assert any('Drift detected' in m for m in caplog.messages)
