import os
import sys
import pandas as pd
import pytest
import logging
import runpy
import warnings

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

# Ignore RuntimeWarning emitted by runpy when module is already imported
warnings.filterwarnings(
    "ignore",
    message=r".*tuning\.hyperparameter_sweep.*",
    category=RuntimeWarning,
)

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
    pd.DataFrame({'profit': [1, -1] * 5}).to_csv(trade_log, index=False)
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
    pd.DataFrame({'profit': [1, -1] * 5}).to_csv(trade_log, index=False)
    hs.run_sweep(str(tmp_path), grid, seed=1, resume=False, trade_log_path=str(trade_log))
    df = pd.read_csv(tmp_path / 'summary.csv')
    assert df.loc[0, 'learning_rate'] == 0.1
    assert df.loc[0, 'depth'] == 6
    assert df.loc[0, 'seed'] == 1


def test_run_sweep_no_log(tmp_path):
    grid = {'p': [1]}
    missing = tmp_path / 'missing.csv'
    with pytest.raises(SystemExit):
        hs.run_sweep(str(tmp_path), grid, trade_log_path=str(missing))
    assert not missing.exists()


def test_parse_args_defaults():
    args = hs.parse_args([])
    assert args.trade_log_path == hs.DEFAULT_TRADE_LOG
    assert args.output_dir == hs.DEFAULT_SWEEP_DIR
    assert args.param_subsample == '0.8,1.0'
    assert args.param_colsample_bylevel == '0.8,1.0'


def test_run_sweep_default_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, 'DEFAULT_SWEEP_DIR', str(tmp_path))

    def dummy_train(output_dir, trade_log_path=None, m1_path=None, seed=0):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': [],
            'metrics': {'acc': 1.0},
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train)
    trade_log = tmp_path / 'log.csv'
    pd.DataFrame({'profit': [1, -1] * 5}).to_csv(trade_log, index=False)
    hs.run_sweep(None, {'lr': [0.1]}, resume=False, trade_log_path=str(trade_log))
    assert (tmp_path / 'best_param.json').exists()


def test_filter_kwargs():
    def fn(a, b=0):
        return a + b

    kwargs = {'a': 1, 'b': 2, 'c': 3}
    filtered = hs._filter_kwargs(fn, kwargs)
    assert filtered == {'a': 1, 'b': 2}


def test_run_sweep_missing_trade_log(tmp_path):
    grid = {'p': [1]}
    with pytest.raises(SystemExit):
        hs.run_sweep(str(tmp_path), grid, trade_log_path=None)


def test_run_sweep_resume_skips(tmp_path, monkeypatch):
    calls = []

    def dummy_train(output_dir, learning_rate=0.1, depth=6, seed=0, trade_log_path=None, m1_path=None):
        calls.append((learning_rate, depth))
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': [],
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train)
    trade_log = tmp_path / 'log.csv'
    pd.DataFrame({'p': [1, 0] * 5}).to_csv(trade_log, index=False)
    summary = tmp_path / 'summary.csv'
    pd.DataFrame([
        {'run_id': 1, 'learning_rate': 0.1, 'depth': 6, 'seed': 1, 'model_path': '', 'features': '', 'metric': None, 'time': 't'},
    ]).to_csv(summary, index=False)
    grid = {'learning_rate': [0.1, 0.2], 'depth': [6]}
    with pytest.raises(SystemExit):
        hs.run_sweep(str(tmp_path), grid, seed=1, resume=True, trade_log_path=str(trade_log))
    assert calls == [(0.2, 6)]
    df = pd.read_csv(summary)
    assert len(df) == 2


def test_run_sweep_resume_missing_column(tmp_path, monkeypatch):
    def dummy_train(output_dir, learning_rate=0.1, depth=6, subsample=1.0, seed=0, trade_log_path=None, m1_path=None):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': [],
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train)
    trade_log = tmp_path / 'log.csv'
    pd.DataFrame({'p': [1, 1] * 5}).to_csv(trade_log, index=False)
    summary = tmp_path / 'summary.csv'
    # summary.csv ไม่มีคอลัมน์ subsample เพื่อทดสอบกรณี missing columns
    pd.DataFrame([
        {'run_id': 1, 'learning_rate': 0.1, 'depth': 6, 'seed': 1, 'model_path': '', 'features': '', 'metric': None, 'time': 't'},
    ]).to_csv(summary, index=False)
    grid = {'learning_rate': [0.1], 'depth': [6], 'subsample': [0.8, 1.0]}
    with pytest.raises(SystemExit):
        hs.run_sweep(str(tmp_path), grid, seed=1, resume=True, trade_log_path=str(trade_log))
    df = pd.read_csv(summary)
    # row เดิม + 2 ชุดใหม่ที่ subsample ต่างกัน
    assert len(df) == 3


def test_run_sweep_no_metric(tmp_path, monkeypatch, caplog):
    def dummy_train(output_dir, trade_log_path=None, m1_path=None, seed=0):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': [],
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train)
    trade_log = tmp_path / 'log.csv'
    pd.DataFrame({'p': [1, 0] * 5}).to_csv(trade_log, index=False)
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({'Open': [1, 2]}).to_csv(m1_path, index=False)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            hs.run_sweep(
                str(tmp_path),
                {'lr': [0.1]},
                resume=False,
                trade_log_path=str(trade_log),
                m1_path=str(m1_path),
            )
    assert 'ไม่มี metric จากผลลัพธ์ sweep' in caplog.text


def test_run_sweep_missing_m1(tmp_path, monkeypatch):
    def dummy_train(output_dir, trade_log_path=None, m1_path=None, seed=0):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': [],
            'metrics': {'acc': 1.0},
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train)
    trade_log = tmp_path / 'log.csv'
    pd.DataFrame({'p': [1, 0] * 5}).to_csv(trade_log, index=False)
    m1_path = tmp_path / 'missing.csv'
    hs.run_sweep(str(tmp_path), {'lr': [0.1]}, trade_log_path=str(trade_log), m1_path=str(m1_path))
    assert m1_path.exists()
    assert (tmp_path / 'best_param.json').exists()


def test_main_passes_args(tmp_path, monkeypatch):
    trade_log = tmp_path / 'log.csv'
    pd.DataFrame({'p': [1, 0] * 5}).to_csv(trade_log, index=False)
    captured = {}

    def fake_run_sweep(output_dir, params_grid, seed, resume, trade_log_path, m1_path):
        captured['output_dir'] = output_dir
        captured['params_grid'] = params_grid
        captured['seed'] = seed
        captured['resume'] = resume
        captured['trade_log_path'] = trade_log_path
        captured['m1_path'] = m1_path

    monkeypatch.setattr(hs, 'run_sweep', fake_run_sweep)
    hs.main([
        '--output_dir', str(tmp_path),
        '--seed', '5',
        '--resume',
        '--param_learning_rate', '0.5',
        '--trade_log_path', str(trade_log),
    ])

    assert captured['output_dir'] == str(tmp_path)
    assert captured['params_grid']['learning_rate'] == [0.5]
    assert captured['seed'] == 5
    assert captured['resume'] is True
    assert captured['trade_log_path'] == str(trade_log)


def test_cli_entrypoint_runs_main(tmp_path, monkeypatch):
    trade_log = tmp_path / 'log.csv'
    pd.DataFrame({'p': [1, 0] * 5}).to_csv(trade_log, index=False)

    def dummy_train(*_, **__):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': [],
            'metrics': {'accuracy': 1.0},
        }

    import src.training as training

    monkeypatch.setattr(training, 'real_train_func', dummy_train)
    monkeypatch.setattr(sys, 'argv', ['hyperparameter_sweep.py', '--output_dir', str(tmp_path), '--trade_log_path', str(trade_log), '--resume'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        runpy.run_module('tuning.hyperparameter_sweep', run_name='__main__')
    assert (tmp_path / 'best_param.json').exists()


def test_parse_args_custom_params():
    args = hs.parse_args([
        '--param_subsample', '0.7,1.0',
        '--param_colsample_bylevel', '0.6,0.9',
    ])
    assert args.param_subsample == '0.7,1.0'
    assert args.param_colsample_bylevel == '0.6,0.9'


def test_default_trade_log_dynamic_lookup(monkeypatch, tmp_path):
    """DEFAULT_TRADE_LOG ควรค้นหาไฟล์ walk-forward ที่มีอยู่แบบไดนามิก"""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    dynamic_log = out_dir / "trade_log_v33_walkforward_custom.csv.gz"
    pd.DataFrame({"profit": [1]}).to_csv(dynamic_log, index=False, compression="gzip")

    import importlib, sys
    monkeypatch.setattr(sys.modules['src.config'].DefaultConfig, 'OUTPUT_DIR', str(out_dir))
    hs_reload = importlib.reload(hs)
    assert hs_reload.DEFAULT_TRADE_LOG == str(dynamic_log)


def test_run_single_trial_skip_on_small_log(monkeypatch, tmp_path, caplog):
    df_log = pd.DataFrame({'profit': range(5)})

    called = {}

    def dummy_train(**kwargs):
        called['yes'] = True
        return {}

    monkeypatch.setattr(hs, 'real_train_func', dummy_train)

    with caplog.at_level(logging.WARNING):
        res = hs._run_single_trial({}, df_log, str(tmp_path), 'log', 'm1')

    assert res is None
    assert 'Skipping trial' in caplog.text
    assert 'yes' not in called


def test_run_single_trial_with_cv(monkeypatch, tmp_path):
    df_log = pd.DataFrame({'profit': [1, -1] * 5})
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({
        'Open': range(10),
        'High': range(10),
        'Low': range(10),
        'Close': range(10)
    }).to_csv(m1_path, index=False)

    def dummy_train(**kwargs):
        return {
            'model_path': {'model': str(tmp_path / 'm.joblib')},
            'features': ['Open'],
            'metrics': {'accuracy': 1.0}
        }

    monkeypatch.setattr(hs, 'real_train_func', dummy_train)

    res = hs._run_single_trial({}, df_log, str(tmp_path), str(tmp_path/'log.csv'), str(m1_path))
    assert 'cv_auc' in res['metrics']


