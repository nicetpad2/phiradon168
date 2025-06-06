import os
from sqlalchemy import create_engine, text

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
