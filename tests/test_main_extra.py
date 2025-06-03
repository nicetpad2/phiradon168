import os
import sys
import json
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

import src.main as main
import src.strategy as strategy


def test_generate_open_signals_basic():
    df = pd.DataFrame({'Close': [1.0, 1.2, 1.1]})
    signals = strategy.generate_open_signals(df)
    assert signals.dtype == np.int8
    assert signals.tolist() == [0, 1, 0]


def test_generate_close_signals_basic():
    df = pd.DataFrame({'Close': [1.0, 0.9, 1.1]})
    signals = strategy.generate_close_signals(df)
    assert signals.tolist() == [0, 1, 0]


def test_precompute_sl_array_length():
    df = pd.DataFrame({'Close': [1, 2, 3]})
    sl = strategy.precompute_sl_array(df)
    assert sl.dtype == np.float64
    assert len(sl) == len(df)


def test_precompute_tp_array_length():
    df = pd.DataFrame({'Close': [1, 2]})
    tp = strategy.precompute_tp_array(df)
    assert tp.dtype == np.float64
    assert len(tp) == len(df)


def test_save_final_data_creates_file(tmp_path):
    df = pd.DataFrame({'A': [1]})
    out_file = tmp_path / 'data.csv'
    main.save_final_data(df, str(out_file))
    assert out_file.exists()


def test_load_features_from_file_returns_dict():
    assert main.load_features_from_file('dummy.json') == {}


def test_setup_output_directory_main(tmp_path):
    path = main.setup_output_directory(str(tmp_path), 'out')
    assert os.path.isdir(path)


def test_drop_nan_rows_no_change():
    df = pd.DataFrame({'A': [1, 2]})
    res = main.drop_nan_rows(df)
    pd.testing.assert_frame_equal(res, df)


def test_convert_to_float32_mixed_dtypes():
    df = pd.DataFrame({'A': [1], 'B': ['x']})
    res = main.convert_to_float32(df)
    assert res['A'].dtype == 'float32'
    assert res['B'].dtype == object


def test_run_initial_backtest_returns_none():
    assert main.run_initial_backtest() is None


def test_ensure_main_features_file_creates(tmp_path):
    path = main.ensure_main_features_file(str(tmp_path))
    file_path = tmp_path / 'features_main.json'
    assert path == str(file_path)
    assert file_path.exists()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data == main.DEFAULT_META_CLASSIFIER_FEATURES


def test_ensure_main_features_file_preserves(tmp_path):
    file_path = tmp_path / 'features_main.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(['A'], f)
    path = main.ensure_main_features_file(str(tmp_path))
    assert path == str(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data == ['A']
