"""Configuration module for phiradon168."""

import logging
import os
from datetime import datetime

# CSV data paths
CSV_PATH_M1 = "/content/drive/MyDrive/Phiradon168/XAUUSD_M1.csv"
CSV_PATH_M15 = "/content/drive/MyDrive/Phiradon168/XAUUSD_M15.csv"

# Logs directory
LOG_DIR = "/content/drive/MyDrive/Phiradon168/logs"

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp for uniqueness
LOG_FILENAME = os.path.join(LOG_DIR, f"gold_ai_{datetime.now():%Y%m%d_%H%M%S}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ],
)

__all__ = [
    "CSV_PATH_M1",
    "CSV_PATH_M15",
    "LOG_DIR",
    "LOG_FILENAME",
]
