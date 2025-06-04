import pandas as pd
from src.feature_analysis import analyze_feature_distribution, detect_low_variance_features


def test_detect_low_variance_features():
    df = pd.DataFrame({'a': [1, 1, 1], 'b': [1, 2, 3]})
    assert detect_low_variance_features(df, ['a', 'b']) == ['a']


def test_analyze_feature_distribution_basic(tmp_path):
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    stats = analyze_feature_distribution(df, ['a'], output_dir=tmp_path)
    assert 'a' in stats
    assert 'mean' in stats['a']
    assert (tmp_path / 'a_hist.png').exists()
    assert (tmp_path / 'a_box.png').exists()
