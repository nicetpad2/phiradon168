import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

import placeholder
import pytest


def test_add_and_multiply():
    assert placeholder.add(2, 3) == 5
    assert placeholder.multiply(4, 5) == 20


def test_required_lines_for_target_basic():
    assert placeholder.required_lines_for_target(8719, 3592, 0.65) == 2075


def test_required_lines_for_target_met():
    # Coverage already above target
    assert placeholder.required_lines_for_target(100, 70, 0.6) == 0


def test_required_lines_for_target_invalid():
    with pytest.raises(ValueError):
        placeholder.required_lines_for_target(0, 0, 0.5)
    with pytest.raises(ValueError):
        placeholder.required_lines_for_target(100, 10, 1.5)
