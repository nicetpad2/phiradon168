import os
import sys
from src.utils.terminal_logger import terminal_logger


def test_terminal_logger_captures_stdout(tmp_path):
    log_file = tmp_path / "t.log"
    with terminal_logger(str(log_file)):
        print("hello world")
        print("oops", file=sys.stderr)
    assert log_file.exists()
    content = log_file.read_text()
    assert "hello world" in content
    assert "oops" in content
