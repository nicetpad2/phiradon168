import os
import sys
import pandas as pd
import numpy as np
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.features as features

import src.data_loader as dl

def test_calculate_trend_zone_returns_neutral_series():
    df = pd.DataFrame({'Close': [1, 2, 3]}, index=pd.RangeIndex(3))
    result = features.calculate_trend_zone(df)
    assert isinstance(result, pd.Series)
    assert result.tolist() == ['NEUTRAL'] * 3


def test_create_session_column_tags_sessions():
    idx = pd.date_range('2024-01-01 00:00', periods=3, freq='6h')
    df = pd.DataFrame({'Open': [1, 2, 3]}, index=idx)
    result = features.create_session_column(df.copy())
    expected = [features.get_session_tag(ts) for ts in idx]
    assert 'session' in result.columns
    assert result['session'].tolist() == expected


def test_create_session_column_handles_none():
    result = features.create_session_column(None)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert 'session' in result.columns


def test_create_session_column_handles_empty_df():
    df = pd.DataFrame({'Open': []})
    result = features.create_session_column(df)
    assert 'session' in result.columns
    assert result.empty


def test_fill_missing_feature_values_replaces_nan():
    df = pd.DataFrame({'A': [1.0, np.nan]})
    result = features.fill_missing_feature_values(df)
    assert result.isna().sum().sum() == 0


def test_load_feature_config_returns_empty_dict(tmp_path):
    path = tmp_path / 'nonexistent.json'
    result = features.load_feature_config(str(path))
    assert result == {}


def test_calculate_ml_features_returns_input_df():
    df = pd.DataFrame({'A': [1, 2]})
    result = features.calculate_ml_features(df)
    pd.testing.assert_frame_equal(result, df)


@pytest.mark.parametrize(
    'context,expected',
    [({'spike_score': 0.7}, 'spike'), ({'cluster': 2}, 'cluster'), ({}, 'main')]
)
def test_select_model_for_trade_variants(context, expected):
    models = {
        'main': {'model': object(), 'features': ['f']},
        'spike': {'model': object(), 'features': ['f']},
        'cluster': {'model': object(), 'features': ['f']},
    }
    result, _ = features.select_model_for_trade(context, models)
    assert result == expected


def test_safe_set_datetime_converts_column():
    df = pd.DataFrame({'Date': ['2024-01-01']})
    dl.safe_set_datetime(df, 0, 'Date', '2024-01-02')
    assert df['Date'].dtype == 'datetime64[ns]'
    assert df.loc[0, 'Date'] == pd.Timestamp('2024-01-02')


def test_safe_set_datetime_index_not_found(caplog):
    df = pd.DataFrame({'Date': ['2024-01-01']})
    dl.safe_set_datetime(df, 5, 'Date', '2024-01-02')
    assert df["Date"].iloc[0] == pd.Timestamp("2024-01-01")

