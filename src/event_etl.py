"""ETL utilities for loading log events into a database."""

# [Patch v5.10.0] New module to store trade events in SQLAlchemy DB

from __future__ import annotations

from datetime import datetime, UTC
import re
from typing import Iterable

from sqlalchemy import (
    Table,
    Column,
    Integer,
    String,
    DateTime,
    MetaData,
    Index,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import logging

# --- SQLAlchemy table setup -------------------------------------------------
metadata = MetaData()

trade_events = Table(
    "trade_events",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("timestamp", DateTime, nullable=False),
    Column("event_type", String(32), nullable=False),
    Column("detail", String, nullable=True),
)
Index("ix_trade_events_timestamp", trade_events.c.timestamp)
Index("ix_trade_events_event_type", trade_events.c.event_type)


def init_db(engine: Engine) -> None:
    """Create the trade_events table if it does not exist."""
    metadata.create_all(engine)


# --- Log parsing ------------------------------------------------------------
EVENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "ATTEMPT",
        re.compile(
            r"Attempting to Open New Order.*?(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})"
        ),
    ),
    ("BLOCK", re.compile(r"BLOCKED.*?(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})?")),
    (
        "EXECUTE",
        re.compile(
            r"Order Opened.*?(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})"
        ),
    ),
    ("KILL_SWITCH", re.compile(r"Kill Switch Activated(?::| )?(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})?")),
]


def _parse_events(lines: Iterable[str]) -> list[dict]:
    events: list[dict] = []
    for line in lines:
        for etype, pattern in EVENT_PATTERNS:
            m = pattern.search(line)
            if m:
                ts_str = m.groupdict().get("time")
                ts = (
                    datetime.fromisoformat(ts_str)
                    if ts_str
                    else datetime.now(UTC)
                )
                events.append({"timestamp": ts, "event_type": etype, "detail": line.strip()})
                break
    return events


def ingest_log_to_db(log_path: str, engine: Engine) -> int:
    """Parse a log file and insert events into the database."""
    with open(log_path, "r", encoding="utf-8") as f:
        events = _parse_events(f)
    if not events:
        return 0
    try:
        with engine.begin() as conn:
            conn.execute(trade_events.insert(), events)
    except SQLAlchemyError as exc:
        logging.getLogger(__name__).error("ingest_log_to_db failed: %s", exc)
        return 0
    return len(events)


__all__ = ["create_engine", "init_db", "ingest_log_to_db", "trade_events"]
