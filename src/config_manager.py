from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigManager:
    """Central configuration handler."""

    data_dir: Path
    model_dir: Path
    log_dir: Path
    data_file_m1: Path
    data_file_m15: Path

    @classmethod
    def load(cls) -> "ConfigManager":
        base_dir = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parent.parent))
        data_dir = Path(os.getenv("DATA_DIR", base_dir / "data"))
        model_dir = Path(os.getenv("MODEL_DIR", base_dir / "models"))
        log_dir = Path(os.getenv("LOG_DIR", base_dir / "logs"))
        data_file_m1 = Path(os.getenv("DATA_FILE_M1", base_dir / "XAUUSD_M1.parquet"))
        data_file_m15 = Path(os.getenv("DATA_FILE_M15", base_dir / "XAUUSD_M15.parquet"))

        for d in (data_dir, model_dir, log_dir):
            d.mkdir(parents=True, exist_ok=True)
        return cls(data_dir, model_dir, log_dir, data_file_m1, data_file_m15)


config = ConfigManager.load()
