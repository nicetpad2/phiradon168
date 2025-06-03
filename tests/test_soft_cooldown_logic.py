import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

SOFT_COOLDOWN_LOOKBACK = 10
# [Patch v5.0.18] Updated threshold to match strategy defaults
SOFT_COOLDOWN_LOSS_COUNT = 6


from cooldown_utils import is_soft_cooldown_triggered


def test_soft_cooldown_requires_lookback():
    pnl_history = [-1] * SOFT_COOLDOWN_LOSS_COUNT
    triggered, _ = is_soft_cooldown_triggered(
        pnl_history, SOFT_COOLDOWN_LOOKBACK, SOFT_COOLDOWN_LOSS_COUNT
    )
    assert not triggered


def test_soft_cooldown_triggers_after_lookback():
    pnl_history = [-1] * SOFT_COOLDOWN_LOOKBACK
    triggered, losses = is_soft_cooldown_triggered(
        pnl_history, SOFT_COOLDOWN_LOOKBACK, SOFT_COOLDOWN_LOSS_COUNT
    )
    assert triggered and losses == SOFT_COOLDOWN_LOOKBACK
