import os
import sys
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src import strategy


def test_attempt_order_blocked_oms(monkeypatch, caplog):
    monkeypatch.setattr(strategy, "OMS_ENABLED", False)
    with caplog.at_level(logging.WARNING):
        ok, reasons = strategy.attempt_order("BUY", 1.0, {})
    assert not ok and reasons == ["OMS_DISABLED"]
    assert "OMS_DISABLED" in caplog.text
    monkeypatch.setattr(strategy, "OMS_ENABLED", True)


def test_attempt_order_executed(monkeypatch, caplog):
    monkeypatch.setattr(strategy, "OMS_ENABLED", True)
    with caplog.at_level(logging.INFO):
        ok, reasons = strategy.attempt_order("BUY", 1.0, {})
    assert ok and reasons == []
    assert "Order Executed" in caplog.text


def test_attempt_order_multiple_reasons(monkeypatch, caplog):
    monkeypatch.setattr(strategy, "OMS_ENABLED", False)
    params = {"kill_switch_active": True, "paper_mode": True}
    with caplog.at_level(logging.WARNING):
        ok, reasons = strategy.attempt_order("SELL", 2.0, params)
    assert not ok
    assert reasons[0] == "OMS_DISABLED"
    assert set(reasons) == {"OMS_DISABLED", "KILL_SWITCH_ACTIVE", "PAPER_MODE_SIMULATION"}
