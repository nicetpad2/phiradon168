import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.data_loader as dl


def test_deduplicate_and_sort_basic():
    df = pd.DataFrame({
        'Date': [20240101, 20240101, 20240101],
        'Timestamp': ['00:00:00', '00:00:00', '00:01:00'],
        'Close': [1, 2, 3]
    })
    res = dl.deduplicate_and_sort(df, subset_cols=['Date', 'Timestamp'])
    assert len(res) == 2
    assert list(res['Timestamp']) == ['00:00:00', '00:01:00']
    assert res.iloc[0]['Close'] == 2


def test_deduplicate_and_sort_missing_columns():
    df = pd.DataFrame({'A': [1, 1]})
    res = dl.deduplicate_and_sort(df, subset_cols=['Date', 'Timestamp'])
    assert len(res) == 2
