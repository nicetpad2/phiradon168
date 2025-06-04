import importlib
from src import strategy, main

def test_kill_switch_default_thresholds():
    assert strategy.DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD == 0.30
    assert strategy.DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD == 10
    assert main.DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD == 0.30
    assert main.DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD == 10

def test_kill_switch_warning_thresholds():
    assert strategy.DEFAULT_KILL_SWITCH_WARNING_MAX_DD_THRESHOLD == 0.25
    assert strategy.DEFAULT_KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD == 7
    assert main.DEFAULT_KILL_SWITCH_WARNING_MAX_DD_THRESHOLD == 0.25
    assert main.DEFAULT_KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD == 7
