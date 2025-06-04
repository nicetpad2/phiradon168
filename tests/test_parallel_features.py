import pandas as pd
from profile_backtest import calculate_features_for_fold, run_parallel_feature_engineering


def test_parallel_feature_engineering():
    df = pd.DataFrame({'Open': [1,2], 'High':[2,3], 'Low':[1,2], 'Close':[2,3]})
    params = [('XAUUSD', df)]
    res_seq = calculate_features_for_fold(params[0])
    res_par = run_parallel_feature_engineering(params, processes=1)[0]
    assert res_seq.equals(res_par)
