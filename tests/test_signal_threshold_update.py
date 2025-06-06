import os
import sys
import logging
from unittest.mock import patch
from dataclasses import dataclass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.adaptive import update_signal_threshold

@dataclass
class DummyParams:
    signal_score_threshold: float = 0.3
    current_fold: int = 1
    profile_name: str = 'Demo'


def test_update_signal_threshold_high():
    params = DummyParams()
    logger = logging.getLogger('src.adaptive')
    with patch.object(logger, 'info') as mock_info:
        th = update_signal_threshold(0.9, params)
    assert th == 0.5
    mock_info.assert_called()


def test_update_signal_threshold_low():
    params = DummyParams()
    logger = logging.getLogger('src.adaptive')
    with patch.object(logger, 'info') as mock_info:
        th = update_signal_threshold(0.1, params)
    assert th == 0.25
    mock_info.assert_called()


def test_update_signal_threshold_invalid():
    params = DummyParams()
    logger = logging.getLogger('src.adaptive')
    with patch.object(logger, 'warning') as mock_warn:
        th = update_signal_threshold('bad', params)
    assert th == params.signal_score_threshold
    mock_warn.assert_called_once()
