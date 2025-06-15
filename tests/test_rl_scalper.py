import numpy as np
from src.adaptive import RLScalperAgent


def test_choose_action_greedy():
    agent = RLScalperAgent(epsilon=0.0)
    state = (1, 0)
    agent.q_table["1|0"] = np.array([0.1, 0.5, 0.2])
    assert agent.choose_action(state) == 1


def test_update_and_epsilon_decay():
    agent = RLScalperAgent(alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.1)
    state = (0,)
    next_state = (1,)
    agent.update(state, 1, 1.0, next_state, done=False)
    assert "0" in agent.q_table
    agent.update(state, 1, 1.0, next_state, done=True)
    assert agent.epsilon == 0.5
    agent.update(state, 1, 1.0, next_state, done=True)
    assert agent.epsilon == 0.25
    agent.update(state, 1, 1.0, next_state, done=True)
    assert agent.epsilon == 0.125
    agent.update(state, 1, 1.0, next_state, done=True)
    assert agent.epsilon == 0.1


def test_save_and_load(tmp_path):
    agent = RLScalperAgent(epsilon=0.0)
    agent.q_table["0"] = np.array([1.0, 2.0, 3.0])
    file_path = tmp_path / "q.json"
    agent.save(file_path)
    new_agent = RLScalperAgent(epsilon=0.0)
    new_agent.load(file_path)
    assert np.allclose(new_agent.q_table["0"], [1.0, 2.0, 3.0])
