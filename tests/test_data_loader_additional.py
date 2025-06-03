import os
import json
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.data_loader as dl

def test_load_app_config_missing(tmp_path):
    conf_path = tmp_path / 'nonexistent.json'
    result = dl.load_app_config(str(conf_path))
    assert result == {}


def test_load_app_config_valid(tmp_path):
    conf_data = {"x": 1}
    conf_file = tmp_path / 'cfg.json'
    conf_file.write_text(json.dumps(conf_data), encoding='utf-8')
    result = dl.load_app_config(str(conf_file))
    assert result == conf_data


def test_safe_set_datetime_basic():
    df = pd.DataFrame(index=[0])
    dl.safe_set_datetime(df, 0, 'Date', '2023-01-02')
    assert df['Date'].dtype == 'datetime64[ns]'
    assert df.loc[0, 'Date'] == pd.Timestamp('2023-01-02')


def test_rsi_returns_nan_when_ta_missing(monkeypatch):
    series = pd.Series([1, 2, 3], dtype='float32')
    monkeypatch.setattr(dl, 'ta', None, raising=False)
    import importlib
    features = importlib.import_module('src.features')
    monkeypatch.setattr(features, 'ta', None, raising=False)
    result = features.rsi(series, period=5)
    assert result.isna().all()

def test_simple_converter_edge_cases():
    assert dl.simple_converter(np.inf) == "Infinity"
    assert dl.simple_converter(-np.inf) == "-Infinity"
    assert dl.simple_converter(pd.NaT) is None
    class Dummy:
        def __str__(self):
            return "dummy"
    assert dl.simple_converter(Dummy()) == "dummy"


def test_safe_get_global_returns_default():
    assert dl.safe_get_global('NON_EXISTENT_VAR', 123) == 123

def test_setup_output_directory_creates(tmp_path):
    base = tmp_path
    result = dl.setup_output_directory(str(base), 'out')
    assert os.path.isdir(result)
