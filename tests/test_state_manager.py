import json
from src.state_manager import StateManager


def test_state_manager_load_and_save(tmp_path):
    state_file = tmp_path / 'state.json'
    manager = StateManager(state_file_path=str(state_file))
    assert manager.get_state('consecutive_losses') == 0

    assert manager.update_state('consecutive_losses', 2)
    manager.save_state()

    manager2 = StateManager(state_file_path=str(state_file))
    assert manager2.get_state('consecutive_losses') == 2


def test_update_state_no_change(tmp_path):
    state_file = tmp_path / 'state.json'
    manager = StateManager(state_file_path=str(state_file))
    assert not manager.update_state('consecutive_losses', 0)

    assert manager.update_state('new_key', 5)
    assert manager.get_state('new_key') == 5
