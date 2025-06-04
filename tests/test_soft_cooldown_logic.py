import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

SOFT_COOLDOWN_LOOKBACK = 15
# [Patch v5.4.6] Adjust defaults for real market conditions
SOFT_COOLDOWN_LOSS_COUNT = 2


from cooldown_utils import is_soft_cooldown_triggered, step_soft_cooldown


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


