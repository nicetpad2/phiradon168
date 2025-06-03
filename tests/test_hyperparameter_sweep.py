import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from src.strategy import run_hyperparameter_sweep

def test_run_hyperparameter_sweep_basic(tmp_path):
    calls = []
    def dummy_train_func(**kwargs):
        calls.append(kwargs)
        return {"model": "path"}, ["f1", "f2"]

    base_params = {"output_dir": str(tmp_path)}
    grid = {"p1": [1, 2], "p2": [0.1, 0.2]}
    results = run_hyperparameter_sweep(base_params, grid, train_func=dummy_train_func)
    assert len(results) == 4
    assert len(calls) == 4
    for res in results:
        assert "model_paths" in res and "features" in res
