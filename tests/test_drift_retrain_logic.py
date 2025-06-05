import pandas as pd
from src.strategy import DriftObserver, DRIFT_WASSERSTEIN_THRESHOLD


def test_needs_retrain_true():
    df_train = pd.DataFrame({'ADX': [0.0] * 20})
    df_test = pd.DataFrame({'ADX': [2.0] * 20})
    obs = DriftObserver(['ADX'])
    obs.analyze_fold(df_train, df_test, 0)
    assert obs.needs_retrain(0, threshold=0.5)


def test_needs_retrain_false():
    df_train = pd.DataFrame({'ADX': [0.0] * 20})
    df_test = pd.DataFrame({'ADX': [0.1] * 20})
    obs = DriftObserver(['ADX'])
    obs.analyze_fold(df_train, df_test, 0)
    assert not obs.needs_retrain(0, threshold=0.5)
