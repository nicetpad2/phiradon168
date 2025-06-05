import os
from pathlib import Path

class Config:
    """Basic configuration loader with type checking."""

    def __init__(self):
        self.DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
        self.MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
        self.LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
        self.NUM_WORKERS = self._parse_int("NUM_WORKERS", 1)
        self.LEARNING_RATE = self._parse_float("LEARNING_RATE", 0.001)
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
