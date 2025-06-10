import pytest
from types import SimpleNamespace
from pathlib import Path
import pandas as pd

from src.utils.auto_train_meta_classifiers import auto_train_meta_classifiers
from src.config import logger


def test_auto_train_meta_classifiers_loads_trade_log(tmp_path, caplog):
    """Should load trade log file when present."""
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    df = pd.DataFrame({"x": [1]})
    df.to_csv(Path(cfg.OUTPUT_DIR) / "trade_log_v32_walkforward.csv.gz", index=False, compression="gzip")
    with caplog.at_level('INFO', logger=logger.name):
        result = auto_train_meta_classifiers(cfg, None)
    assert result is None
    assert any("Patch v6.4.2" in m and "Loading trade log" in m for m in caplog.messages)


def test_auto_train_meta_classifiers_missing_trade_log(tmp_path, caplog):
    """Should log an error when trade log is absent."""
    cfg = SimpleNamespace(OUTPUT_DIR=str(tmp_path))
    with caplog.at_level('ERROR', logger=logger.name):
        result = auto_train_meta_classifiers(cfg, None)
    assert result is None
    assert any("Patch v6.4.2" in m and "not found" in m for m in caplog.messages)
