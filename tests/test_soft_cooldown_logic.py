import pandas as pd

SOFT_COOLDOWN_LOOKBACK = 10
SOFT_COOLDOWN_LOSS_COUNT = 3


def check_can_open(pnls):
    can_open = True
    if can_open and len(pnls) >= SOFT_COOLDOWN_LOOKBACK:
        recent_losses = sum(1 for p in pnls[-SOFT_COOLDOWN_LOOKBACK:] if p < 0)
        if recent_losses >= SOFT_COOLDOWN_LOSS_COUNT:
            can_open = False
    return can_open


def test_soft_cooldown_requires_lookback():
    pnl_history = [-1] * SOFT_COOLDOWN_LOSS_COUNT
    assert check_can_open(pnl_history)


def test_soft_cooldown_triggers_after_lookback():
    pnl_history = [-1] * SOFT_COOLDOWN_LOOKBACK
    assert not check_can_open(pnl_history)
