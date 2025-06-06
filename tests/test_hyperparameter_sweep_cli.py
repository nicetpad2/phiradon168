import os
import sys
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import tuning.hyperparameter_sweep as hs


def test_parse_csv_list():
    assert hs._parse_csv_list('1,2,3', int) == [1, 2, 3]
    assert hs._parse_csv_list('0.1,0.2', float) == [0.1, 0.2]


def test_parse_multi_params():
    class Dummy:
        def __init__(self):
            self.param_a = '1,2'
            self.param_b = '0.1,0.2'

    params = hs._parse_multi_params(Dummy())
    assert params == {'a': [1, 2], 'b': [0.1, 0.2]}


def test_run_sweep_basic(tmp_path, monkeypatch):
    def dummy_train_func(output_dir, learning_rate=0.01, depth=6, l2_leaf_reg=None, seed=0, trade_log_path=None, m1_path=None):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': ['f'],
            'metrics': {'accuracy': 1.0},
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train_func)
    grid = {'learning_rate': [0.1], 'depth': [6]}
    trade_log = tmp_path / 'log.csv'
    pd.DataFrame({'profit': [1, -1]}).to_csv(trade_log, index=False)
    hs.run_sweep(str(tmp_path), grid, seed=1, resume=False, trade_log_path=str(trade_log))
    df = pd.read_csv(tmp_path / 'summary.csv')
    assert df.loc[0, 'learning_rate'] == 0.1
    assert df.loc[0, 'depth'] == 6
    assert df.loc[0, 'seed'] == 1
    assert 'metric' in df.columns
    assert os.path.exists(tmp_path / 'best_param.json')


def test_run_sweep_filters_unknown_params(tmp_path, monkeypatch):
    def dummy_train_func(output_dir, learning_rate=0.01, trade_log_path=None, m1_path=None):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': ['f'],
            'metrics': {'accuracy': 1.0},
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train_func)
    grid = {'learning_rate': [0.1], 'depth': [6]}
    trade_log = tmp_path / 'log.csv'
    pd.DataFrame({'profit': [1]}).to_csv(trade_log, index=False)
    hs.run_sweep(str(tmp_path), grid, seed=1, resume=False, trade_log_path=str(trade_log))
    df = pd.read_csv(tmp_path / 'summary.csv')
    assert df.loc[0, 'learning_rate'] == 0.1
    assert df.loc[0, 'depth'] == 6
    assert df.loc[0, 'seed'] == 1


def test_run_sweep_no_log(tmp_path):
    grid = {'p': [1]}
    with pytest.raises(SystemExit):
        hs.run_sweep(str(tmp_path), grid, trade_log_path=str(tmp_path/'missing.csv'))


def test_parse_args_defaults():
    args = hs.parse_args([])
    assert args.trade_log_path == hs.DEFAULT_TRADE_LOG


