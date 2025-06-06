# [Patch v5.9.13] Additional negative tests for signal classifier
import os
import sys
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import src.signal_classifier as sc


def test_create_label_from_backtest_errors():
    with pytest.raises(ValueError):
        sc.create_label_from_backtest(pd.DataFrame())
    with pytest.raises(KeyError):
        sc.create_label_from_backtest(pd.DataFrame({'Close':[1,2,3]}))


def test_add_basic_features_missing_columns():
    with pytest.raises(ValueError):
        sc.add_basic_features(pd.DataFrame())
    with pytest.raises(KeyError):
        sc.add_basic_features(pd.DataFrame({'Close':[1], 'High':[1], 'Low':[1]}))
