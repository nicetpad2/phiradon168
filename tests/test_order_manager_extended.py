import os
import sys
from datetime import datetime, timedelta
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from strategy import (
    OrderManager,
    RiskManager,
    OrderStatusOM,
    OrderStatusRM,
    calculate_position_size,
)
from src.utils import Settings


def test_order_manager_cooldown_block():
    settings = Settings(cooldown_secs=2, kill_switch_pct=1.0)
    om = OrderManager(settings=settings)
    now = datetime.utcnow()
    assert om.place_order({}, now) is OrderStatusOM.OPEN
    status = om.place_order({}, now + timedelta(seconds=1))
    assert status is OrderStatusOM.BLOCKED_COOLDOWN


def test_order_manager_kill_switch():
    settings = Settings(cooldown_secs=0, kill_switch_pct=0.5)
    om = OrderManager(settings=settings)
    now = datetime.utcnow()
    assert om.place_order({}, now) is OrderStatusOM.OPEN
    om.update_drawdown(0.6)
    status = om.place_order({}, now + timedelta(seconds=1))
    assert status is OrderStatusOM.KILL_SWITCH


def test_risk_manager_kill_switch():
    settings = Settings(kill_switch_pct=0.3)
    rm = RiskManager(settings=settings)
    rm.update_drawdown(0.4)
    assert rm.check_kill_switch() is OrderStatusRM.KILL_SWITCH
    rm.update_drawdown(0.1)
    assert rm.check_kill_switch() is OrderStatusRM.OPEN


def test_calculate_position_size_errors():
    with pytest.raises(ValueError):
        calculate_position_size(0, 0.1, 10)

