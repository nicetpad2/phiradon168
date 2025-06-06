import os
import sys
import logging
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
    logger.setLevel(logging.INFO)
    logger.propagate = False
    records = []
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.emit = lambda record: records.append(record.getMessage())
    logger.addHandler(handler)
    try:
        th = update_signal_threshold(0.9, params)
    finally:
        logger.removeHandler(handler)
    assert th == 0.5
    assert any('[Adaptive] Threshold changed' in m for m in records)


def test_update_signal_threshold_low():
    params = DummyParams()
    logger = logging.getLogger('src.adaptive')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    records = []
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.emit = lambda record: records.append(record.getMessage())
    logger.addHandler(handler)
    try:
        th = update_signal_threshold(0.1, params)
    finally:
        logger.removeHandler(handler)
    assert th == 0.25
    assert any('[Adaptive] Threshold changed' in m for m in records)
