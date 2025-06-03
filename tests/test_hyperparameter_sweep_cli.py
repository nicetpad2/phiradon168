import os
import sys
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

import hyperparameter_sweep as hs


def test_parse_csv_list():
    assert hs._parse_csv_list('1,2,3', int) == [1, 2, 3]
    assert hs._parse_csv_list('0.1,0.2', float) == [0.1, 0.2]


def test_run_sweep_basic(tmp_path, monkeypatch):
    def dummy_train_func(output_dir, learning_rate=0.01, depth=6):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': ['f'],
            'metrics': {'acc': 1.0},
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train_func)
    hs.run_sweep(str(tmp_path), [0.1], [6])
    df = pd.read_csv(tmp_path / 'summary.csv')
    assert df.loc[0, 'learning_rate'] == 0.1
    assert df.loc[0, 'depth'] == 6


def test_parse_grid_args():
    entries = ['p1=1,2', 'p2=0.1,0.2']
    grid = hs._parse_grid_args(entries)
    assert grid == {'p1': [1, 2], 'p2': [0.1, 0.2]}


def test_run_general_sweep(tmp_path, monkeypatch):
    def dummy_train_func(**kwargs):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': ['f'],
            'metrics': {'acc': 1.0},
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train_func)
    grid = {'learning_rate': [0.1], 'depth': [6], 'p1': [1]}
    hs.run_general_sweep(str(tmp_path), grid)
    df = pd.read_csv(tmp_path / 'summary.csv')
    assert set(df.columns) >= {'learning_rate', 'depth', 'p1'}
