import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from src.adaptive import adaptive_sl_tp, adaptive_risk, log_best_params


def test_adaptive_sl_tp_high_vol():
    sl, tp = adaptive_sl_tp(2.0, 1.0, base_sl=2.0, base_tp=1.8)
    assert sl > 2.0 and tp > 1.8


def test_adaptive_risk_reduce():
    risk = adaptive_risk(80, 100, base_risk=0.01, dd_threshold=0.1)
    assert risk < 0.01


def test_log_best_params(tmp_path):
    path = log_best_params({"a": 1}, 0, tmp_path)
    assert path is not None and os.path.isfile(path)
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    assert data["a"] == 1
