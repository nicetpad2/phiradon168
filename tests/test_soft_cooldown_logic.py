import os
import sys
import logging
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))

SOFT_COOLDOWN_LOOKBACK = 15
# [Patch v5.4.6] Adjust defaults for real market conditions
SOFT_COOLDOWN_LOSS_COUNT = 2


from cooldown_utils import is_soft_cooldown_triggered, step_soft_cooldown, CooldownManager


def test_soft_cooldown_triggers_with_fewer_trades():
    pnl_history = [-1] * SOFT_COOLDOWN_LOSS_COUNT
    triggered, losses = is_soft_cooldown_triggered(
        pnl_history, SOFT_COOLDOWN_LOOKBACK, SOFT_COOLDOWN_LOSS_COUNT
    )
    assert triggered and losses == SOFT_COOLDOWN_LOSS_COUNT


def test_soft_cooldown_triggers_after_lookback():
    pnl_history = [-1] * SOFT_COOLDOWN_LOOKBACK
    triggered, losses = is_soft_cooldown_triggered(
        pnl_history, SOFT_COOLDOWN_LOOKBACK, SOFT_COOLDOWN_LOSS_COUNT
    )
    assert triggered and losses == SOFT_COOLDOWN_LOOKBACK


def test_step_soft_cooldown():
    assert step_soft_cooldown(5) == 4
    assert step_soft_cooldown(1) == 0
    assert step_soft_cooldown(0) == 0
    assert step_soft_cooldown(10, step=5) == 5
    assert step_soft_cooldown(4, step=5) == 0


def test_step_soft_cooldown_invalid():
    with pytest.raises(TypeError):
        step_soft_cooldown("5")


def test_cooldown_manager_flow(caplog):
    manager = CooldownManager(loss_threshold=2, cooldown_period=3)
    with caplog.at_level(logging.INFO):
        manager.record_loss()
        triggered = manager.record_loss()
        assert triggered and manager.in_cooldown
        assert "Entering soft cooldown" in caplog.text
    manager.step(); manager.step(); manager.step()
    assert not manager.in_cooldown


def test_soft_cooldown_same_side():
    pnls = [-1] * 8
    sides = ["BUY"] * 8
    triggered, losses = is_soft_cooldown_triggered(
        pnls,
        15,
        8,
        sides,
        "BUY",
    )
    assert triggered and losses == 8


