import pandas as pd
import warnings
from src import feature_analysis
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


def test_feature_analysis_handles_missing_feature(tmp_path, caplog):
    df = pd.DataFrame({'b': [1, 2, 3]})
    with caplog.at_level('WARNING'):
        stats = analyze_feature_distribution(df, ['a'], output_dir=tmp_path)
    assert stats == {}
    assert 'not found' in caplog.text


def test_detect_low_variance_features_handles_missing():
    df = pd.DataFrame({'b': [1, 2, 3]})
    result = detect_low_variance_features(df, ['a', 'b'])
    assert result == []


def test_feature_analysis_main_smoke(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stats, low_var = feature_analysis.main(sample_rows=5)
    assert isinstance(stats, dict)
    assert isinstance(low_var, list)
