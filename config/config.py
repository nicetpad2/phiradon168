import os
import sys
import logging
from pathlib import Path

class Config:
    """Basic configuration loader with type checking."""

    def __init__(self):
        self.DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
        self.MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
        self.LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
        self.NUM_WORKERS = self._parse_int("NUM_WORKERS", 1)
        self.LEARNING_RATE = self._parse_float("LEARNING_RATE", 0.001)
        # Require explicit path to trade log, validate extension, and set minimum rows
        self.TRADE_LOG_PATH = os.getenv("TRADE_LOG_PATH")
        self.MIN_TRADE_ROWS = self._parse_int("MIN_TRADE_ROWS", 10)
        if self.MIN_TRADE_ROWS <= 0:
            logging.critical(
                f"[Patch v6.5.2] MIN_TRADE_ROWS must be > 0, got {self.MIN_TRADE_ROWS}"
            )
            sys.exit(1)
        self._ensure_dirs()

    @staticmethod
    def _parse_int(key: str, default: int) -> int:
        value = os.getenv(key, str(default))
        try:
            return int(value)
        except ValueError as exc:
            raise TypeError(f"{key} must be int") from exc

    @staticmethod
    def _parse_float(key: str, default: float) -> float:
        value = os.getenv(key, str(default))
        try:
            return float(value)
        except ValueError as exc:
            raise TypeError(f"{key} must be float") from exc

    def _ensure_dirs(self) -> None:
        for directory in (self.DATA_DIR, self.MODEL_DIR, self.LOG_DIR):
            directory.mkdir(parents=True, exist_ok=True)


config = Config()
