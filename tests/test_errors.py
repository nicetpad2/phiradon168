import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.errors import log_and_raise
import pytest

def test_log_and_raise(tmp_path):
    log_file = tmp_path / 'e.log'
    with pytest.raises(ValueError):
        try:
            raise ValueError('x')
        except ValueError as e:
            log_and_raise(e, str(log_file))
    assert log_file.exists()
    content = log_file.read_text()
    assert 'ValueError' in content
