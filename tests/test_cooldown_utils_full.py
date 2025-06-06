import os
import sys
import logging
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))

from cooldown_utils import (
    CooldownManager,
    CooldownState,
    is_soft_cooldown_triggered,
    step_soft_cooldown,
    update_losses,
    update_drawdown,
    should_enter_cooldown,
    enter_cooldown,
    should_warn_drawdown,
)


def test_manager_invalid_init():
    with pytest.raises(TypeError):
        CooldownManager(loss_threshold='3')
    with pytest.raises(TypeError):
        CooldownManager(3, '5')


def test_manager_loss_type_error():
    manager = CooldownManager()
    manager.losses_count = 'two'
    with pytest.raises(TypeError):
        manager.record_loss()


def test_manager_reset_flow(caplog):
    manager = CooldownManager(loss_threshold=1, cooldown_period=1)
    with caplog.at_level(logging.INFO):
        manager.record_loss()
        assert manager.in_cooldown
        manager.step()
        assert not manager.in_cooldown
        assert 'Exiting soft cooldown' in caplog.text
    manager.record_win()
    assert manager.losses_count == 0
    manager.reset()
    assert manager.losses_count == 0 and manager.cooldown_counter == 0 and not manager.in_cooldown


def test_is_soft_cooldown_errors():
    with pytest.raises(TypeError):
        is_soft_cooldown_triggered([1], '5', 2)
    with pytest.raises(TypeError):
        is_soft_cooldown_triggered([1], 5, '2')


def test_is_soft_cooldown_empty():
    triggered, losses = is_soft_cooldown_triggered([])
    assert not triggered and losses == 0


def test_is_soft_cooldown_side_filter():
    pnls = [-1, 1, -1, -1]
    sides = ['BUY', 'SELL', 'BUY', 'BUY']
    triggered, losses = is_soft_cooldown_triggered(pnls, 4, 2, sides, 'BUY')
    assert triggered and losses == 3


def test_step_soft_cooldown_invalid_step():
    with pytest.raises(TypeError):
        step_soft_cooldown(5, step='1')


def test_update_losses_paths():
    state = CooldownState()
    update_losses(state, -1)
    assert state.consecutive_losses == 1
    update_losses(state, 1)
    assert state.consecutive_losses == 0


def test_update_drawdown_invalid():
    state = CooldownState()
    with pytest.raises(TypeError):
        update_drawdown(state, '0.1')


def test_should_enter_cooldown():
    state = CooldownState(consecutive_losses=2, drawdown_pct=0.1)
    assert should_enter_cooldown(state, 2, 0.2)
    assert should_enter_cooldown(state, 3, 0.05)
    with pytest.raises(TypeError):
        should_enter_cooldown(state, '2', 0.1)
    with pytest.raises(TypeError):
        should_enter_cooldown(state, 2, '0.1')


def test_enter_cooldown_invalid():
    state = CooldownState()
    with pytest.raises(TypeError):
        enter_cooldown(state, '5')


def test_should_warn_drawdown_invalid():
    state = CooldownState()
    with pytest.raises(TypeError):
        should_warn_drawdown(state, '0.1')
