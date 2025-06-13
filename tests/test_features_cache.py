import pandas as pd
import src.features as features

def test_load_or_engineer_uses_cache(monkeypatch, tmp_path):
    cached = pd.DataFrame({'A':[1,2]})
    path = tmp_path / 'cache.csv'
    features.save_features(cached, str(path), fmt='csv')

    called = False
    def fake_engineer(df, *a, **k):
        nonlocal called
        called = True
        return pd.DataFrame()
    monkeypatch.setattr(features, 'engineer_m1_features', fake_engineer)

    df_raw = pd.DataFrame({'Open':[1], 'High':[1], 'Low':[1], 'Close':[1]})
    result = features.load_or_engineer_m1_features(df_raw, cache_path=str(path), fmt='csv')
    assert not called
    pd.testing.assert_frame_equal(result, cached)


def test_load_or_engineer_generates_and_saves(monkeypatch, tmp_path):
    path = tmp_path / 'cache.csv'
    engineered = pd.DataFrame({'B':[3]})
    def fake_engineer(df, *a, **k):
        return engineered
    monkeypatch.setattr(features, 'engineer_m1_features', fake_engineer)

    df_raw = pd.DataFrame({'Open':[1], 'High':[1], 'Low':[1], 'Close':[1]})
    result = features.load_or_engineer_m1_features(df_raw, cache_path=str(path), fmt='csv')
    assert path.exists()
    pd.testing.assert_frame_equal(result, engineered)
