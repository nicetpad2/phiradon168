import sys
import os
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.utils.leakage import hash_df, timestamp_split, assert_no_overlap


def test_hash_df_consistent():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    h1 = hash_df(df)
    h2 = hash_df(df.copy())
    assert h1 == h2


def test_timestamp_split_and_no_overlap():
    idx = pd.date_range('2024-01-01', periods=6, freq='D')
    df = pd.DataFrame({'x': range(6)}, index=idx)
    train, test = timestamp_split(df, '2024-01-01', '2024-01-03', '2024-01-04', '2024-01-06')
    assert len(train) == 3
    assert len(test) == 3
    assert_no_overlap(train, test)


def test_assert_no_overlap_raises():
    idx = pd.date_range('2024-01-01', periods=4, freq='D')
    df = pd.DataFrame({'x': range(4)}, index=idx)
    train = df.iloc[:3]
    test = df.iloc[2:]
    with pytest.raises(ValueError):
        assert_no_overlap(train, test)
