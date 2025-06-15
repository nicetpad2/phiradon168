import os
import sys
import pandas as pd
import logging
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from tuning.hyperparameter_sweep import load_walk_forward_trade_log


def test_load_walk_forward_trade_log_success(tmp_path, caplog):
    log_file = tmp_path / "log.csv"
    pd.DataFrame({'entry_time': [1], 'exit_time': [2], 'pnl': [0.5]}).to_csv(log_file, index=False)
    with caplog.at_level(logging.INFO):
        df = load_walk_forward_trade_log(str(log_file), logging.getLogger("t"))
    assert not df.empty
    assert "กำลังโหลด trade log" in caplog.text


def test_load_walk_forward_trade_log_missing(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_walk_forward_trade_log(str(missing), logging.getLogger("t"))
