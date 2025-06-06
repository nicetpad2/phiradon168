import math
from src import strategy


def test_check_forced_trigger_true():
    bars = strategy.FORCED_ENTRY_BAR_THRESHOLD
    score = strategy.FORCED_ENTRY_MIN_SIGNAL_SCORE + 0.1
    triggered, info = strategy.check_forced_trigger(bars, score)
    assert triggered is True
    assert info["bars_since_last"] == bars
    assert math.isclose(info["score"], score)


def test_check_forced_trigger_false_score():
    bars = strategy.FORCED_ENTRY_BAR_THRESHOLD
    score = strategy.FORCED_ENTRY_MIN_SIGNAL_SCORE - 0.1
    triggered, _ = strategy.check_forced_trigger(bars, score)
    assert not triggered


def test_check_forced_trigger_false_bars():
    bars = strategy.FORCED_ENTRY_BAR_THRESHOLD - 1
    score = strategy.FORCED_ENTRY_MIN_SIGNAL_SCORE + 0.5
    triggered, _ = strategy.check_forced_trigger(bars, score)
    assert not triggered
