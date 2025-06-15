import time
from datetime import timedelta
import os
import sys

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(1, os.path.join(ROOT_DIR, 'src'))

from src.auth_manager import AuthManager, SESSION_TIMEOUT


def test_register_and_authenticate(tmp_path, monkeypatch):
    user_file = tmp_path / 'users.json'
    auth = AuthManager(user_file=str(user_file))
    auth.register('alice', 'secret')
    session = auth.authenticate('alice', 'secret')
    assert auth.validate_session(session.token)
    auth.logout(session.token)
    assert not auth.validate_session(session.token)


def test_authenticate_invalid(tmp_path):
    auth = AuthManager(user_file=str(tmp_path / 'u.json'))
    with pytest.raises(ValueError):
        auth.authenticate('bob', 'bad')


def test_session_expiry(tmp_path, monkeypatch):
    user_file = tmp_path / 'users.json'
    auth = AuthManager(user_file=str(user_file))
    auth.register('bob', 'pass')
    monkeypatch.setattr('src.auth_manager.SESSION_TIMEOUT', timedelta(seconds=1))
    session = auth.authenticate('bob', 'pass')
    assert auth.validate_session(session.token)
    time.sleep(1.1)
    assert not auth.validate_session(session.token)
