import logging
import os
import sys
from contextlib import contextmanager


class _StreamLogger:
    """File-like object that redirects writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.level = level

    def write(self, message: str) -> None:  # pragma: no cover - trivial
        message = message.rstrip()
        if not message:
            return
        for line in message.splitlines():
            self.logger.log(self.level, line)

    def flush(self) -> None:  # pragma: no cover - compatibility
        pass


@contextmanager
def terminal_logger(log_file: str):
    """Redirect stdout/stderr to ``log_file`` using the logging system."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("terminal")
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _StreamLogger(logger, logging.INFO)
    sys.stderr = _StreamLogger(logger, logging.ERROR)
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        logger.removeHandler(handler)
        handler.close()
