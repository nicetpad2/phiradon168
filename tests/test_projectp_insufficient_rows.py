import logging
from pathlib import Path
import pandas as pd
import ProjectP
import sys
import pytest
from src.utils.errors import PipelineError


def test_insufficient_rows_logs_warning(monkeypatch, tmp_path, caplog):
    csv_path = tmp_path / "log.csv"
    csv_path.write_text("")

    orig_read_csv = ProjectP.pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        if Path(path) == csv_path:
            return pd.DataFrame()
        return orig_read_csv(path, *args, **kwargs)

    test_logger = logging.getLogger("test_logger")
    test_logger.setLevel(logging.WARNING)
    monkeypatch.setattr(ProjectP, "logger", test_logger)
    import src.trade_log_pipeline as tlp
    monkeypatch.setattr(tlp, "logger", test_logger)
    import src.trade_log_pipeline as tlp
    monkeypatch.setattr(tlp, "logger", test_logger)
    monkeypatch.setattr(ProjectP.pd, "read_csv", fake_read_csv)

    def fake_engine(_):
        return pd.DataFrame({"pnl": [1.0]})

    import types

    monkeypatch.setitem(
        sys.modules,
        "backtest_engine",
        types.SimpleNamespace(run_backtest_engine=fake_engine),
    )

    monkeypatch.setattr(ProjectP, "load_features", lambda p: pd.DataFrame())

    with caplog.at_level(logging.INFO, logger="test_logger"):
        df = ProjectP.load_trade_log(str(csv_path), min_rows=10)
    assert not df.empty
    assert any(
        "Generated trade log with" in rec.getMessage() for rec in caplog.records
    )


def test_regeneration_empty_dataframe(monkeypatch, tmp_path, caplog):
    csv_path = tmp_path / "log.csv"
    csv_path.write_text("timestamp,price\n1,100")

    test_logger = logging.getLogger("test_logger")
    test_logger.setLevel(logging.WARNING)
    monkeypatch.setattr(ProjectP, "logger", test_logger)

    def fake_engine(_):
        return pd.DataFrame()

    import types

    monkeypatch.setitem(
        sys.modules,
        "backtest_engine",
        types.SimpleNamespace(run_backtest_engine=fake_engine),
    )
    monkeypatch.setattr(ProjectP, "load_features", lambda p: pd.DataFrame())

    with pytest.raises(PipelineError):
        ProjectP.load_trade_log(str(csv_path), min_rows=5)



