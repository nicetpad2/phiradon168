import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging

from src.event_etl import init_db, ingest_log_to_db

SAMPLE_LOG = """
INFO:root: Attempting to Open New Order (Standard) for SELL at 2023-01-01 10:00:00+00:00...
WARNING:root: BLOCKED: Margin check failed at 2023-01-01 10:00:01+00:00
INFO:root: Order Opened at 2023-01-01 10:00:02+00:00
ERROR:root: Kill Switch Activated 2023-01-01 10:05:00+00:00
"""


def test_ingest_log_to_db(tmp_path):
    db_path = tmp_path / "db.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    init_db(engine)
    log_file = tmp_path / "sample.log"
    log_file.write_text(SAMPLE_LOG)
    count = ingest_log_to_db(str(log_file), engine)
    assert count == 4
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT event_type FROM trade_events ORDER BY id")).fetchall()
    assert [r[0] for r in rows] == ["ATTEMPT", "BLOCK", "EXECUTE", "KILL_SWITCH"]


def test_ingest_log_to_db_empty(tmp_path):
    """Should return 0 when log file has no recognized events."""
    db_path = tmp_path / "db.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    init_db(engine)
    log_file = tmp_path / "empty.log"
    log_file.write_text("INFO:root: nothing to see here")
    count = ingest_log_to_db(str(log_file), engine)
    assert count == 0
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM trade_events")).scalar()
    assert result == 0


def test_ingest_log_to_db_exception(tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "db.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    init_db(engine)
    log_file = tmp_path / "sample.log"
    log_file.write_text(SAMPLE_LOG)

    class DummyCtx:
        def __enter__(self):
            raise SQLAlchemyError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(engine, "begin", lambda: DummyCtx())
    with caplog.at_level(logging.ERROR):
        count = ingest_log_to_db(str(log_file), engine)
    assert count == 0
    assert "ingest_log_to_db failed" in caplog.text
