# pragma: no cover
# === START OF PART 1/12 ===

# ==============================================================================
# กำหนดค่า default constants ที่ทดสอบโดย tests/test_config_defaults.py
# ให้เป็น module-level เสมอ (เพื่อหลีกเลี่ยงกรณี pytest import ไม่ครบ)
# ==============================================================================
MIN_SIGNAL_SCORE_ENTRY = 0.3
M15_TREND_RSI_UP = 60
M15_TREND_RSI_DOWN = 40
FORCED_ENTRY_MIN_GAIN_Z_ABS = 0.5
FORCED_ENTRY_ALLOWED_REGIMES = [
    "Normal", "Breakout", "StrongTrend", "Reversal",
    "Pullback", "InsideBar", "Choppy"
]
ENABLE_SOFT_COOLDOWN = True
ADAPTIVE_SIGNAL_SCORE_QUANTILE = 0.4
REENTRY_MIN_PROBA_THRESH = 0.40
OMS_ENABLED = True
OMS_DEFAULT = True
PAPER_MODE = False
POST_TRADE_COOLDOWN_BARS = 2

# [Patch] Expose simple signal toggles for strategy modules
USE_MACD_SIGNALS = True   # Enable MACD conditions in simple signal functions
USE_RSI_SIGNALS = True    # Enable RSI conditions in simple signal functions

# [Patch v5.9.3] Default hyperparameters used in training
LEARNING_RATE = 0.01
DEPTH = 6
L2_LEAF_REG = None

# ==============================================================================
# ป้องกันกรณีที่ pytest import แค่ SimpleNamespace เดิม (fallback) โดยตรวจสอบสภาพแวดล้อม
# ==============================================================================
import sys

# -*- coding: utf-8 -*-
# <<< เพิ่ม Encoding declaration สำหรับอักษรไทย >>>

# ==============================================================================
# === PART 1: Setup & Configuration (v4.8.4) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added basic docstrings/comments >>>
# <<< MODIFIED v4.8.1: Updated versioning for comprehensive fixes based on prompt >>>
# <<< MODIFIED v4.8.4: Updated paths for Colab/VPS compatibility and versioning >>>
import subprocess
import importlib
import sys
# Imports
import os
import time
import warnings
import atexit
import json
import math
import random
from collections import Counter, defaultdict
from joblib import load, dump as joblib_dump
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
# [Patch v6.5.16] Disable auto-installation of libraries by default
# แนะนำติดตั้ง dependencies ด้วย `pip install -r requirements.txt`
# เปิดใช้ AUTO_INSTALL_LIBS=True เฉพาะในสภาพแวดล้อมพัฒนา
AUTO_INSTALL_LIBS = False  # If True, attempt to auto-install missing libraries
# อ่านเวอร์ชันจากไฟล์ VERSION
VERSION_FILE = os.path.join(os.path.dirname(__file__), '..', 'VERSION')
with open(VERSION_FILE, 'r', encoding='utf-8') as vf:
    __version__ = vf.read().strip()
from pathlib import Path
import pathlib
# [Patch v6.3.1] Register module as 'config' for reload compatibility
if 'src.config' not in sys.modules:
    sys.modules['src.config'] = sys.modules[__name__]

class _ReloadSpecProxy:
    """Proxy __spec__ to ensure `src.config` is registered before reload."""

    def __init__(self, real_spec, module):
        self._spec = real_spec
        self._module = module

    @property
    def name(self):
        if self._spec.name not in sys.modules:
            sys.modules[self._spec.name] = self._module
        return self._spec.name

    def __getattr__(self, attr):
        return getattr(self._spec, attr)

__spec__ = _ReloadSpecProxy(__spec__, sys.modules.get(__name__))
# [Patch v5.9.1] Unified output directory constant
OUTPUT_DIR = Path(__file__).parent.parent / "output_default"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# [Patch v6.2.3] Define models directory and auto threshold flag
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
AUTO_THRESHOLD_TUNING = os.getenv("AUTO_THRESHOLD_TUNING", "False").lower() in ("true", "1", "yes")

# [Patch v6.2.1] Define default data directory and naming for walk-forward data
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Defaults for data file naming
SYMBOL = globals().get("SYMBOL", "XAUUSD")
TIMEFRAME = globals().get("TIMEFRAME", "M1")

# [Patch v6.2.1] Default hyperparameters to prevent missing attribute warnings
for _attr in [
    "subsample",
    "colsample_bylevel",
    "bagging_temperature",
    "random_strength",
    "seed",
]:
    if _attr not in globals():
        globals()[_attr] = None
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder # Added OrdinalEncoder back as it might be used by some logic
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    log_loss,
)
from scipy.stats import ttest_ind, wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from IPython import get_ipython
import shutil
import gzip
import requests  # For Font Download
from src.utils import get_env_float, load_settings, log_settings

import logging

# [Patch] Ensure a module-level logger is always available for imports
logger = logging.getLogger('NiceGold')
logger.setLevel(logging.INFO)

# Load runtime settings and log them
SETTINGS = load_settings()
log_settings(SETTINGS, logger)

# -----------------------------------------------------------------------------
# Fallback defaults for key constants used across the project.
# These are defined up-front so that even when pytest imports this module in a
# restricted environment (or with partial stubs such as SimpleNamespace), tests
# like ``tests/test_config_defaults.py`` can rely on these attributes being
# present. They may be re-assigned later in the file based on environment
# variables or additional logic, but the values below guarantee sane defaults at
# module import time.
# -----------------------------------------------------------------------------
MIN_SIGNAL_SCORE_ENTRY = 0.3
M15_TREND_RSI_UP = 60
M15_TREND_RSI_DOWN = 40
FORCED_ENTRY_MIN_GAIN_Z_ABS = 0.5
FORCED_ENTRY_ALLOWED_REGIMES = [
    "Normal",
    "Breakout",
    "StrongTrend",
    "Reversal",
    "InsideBar",
    "Choppy",
]
ENABLE_SOFT_COOLDOWN = True
ADAPTIVE_SIGNAL_SCORE_QUANTILE = 0.4
REENTRY_MIN_PROBA_THRESH = 0.40
OMS_DEFAULT = True
OMS_ENABLED = True
PAPER_MODE = False
POST_TRADE_COOLDOWN_BARS = 2

# --- Logging Configuration ---
# กำหนดค่าพื้นฐานสำหรับการ Logging
# จัดเก็บไฟล์ log ลงในโฟลเดอร์ย่อยตามวันที่และ fold
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_DATE = datetime.now().strftime('%Y-%m-%d')
FOLD_ID = os.getenv('FOLD_ID', 'fold0')
LOG_DIR = os.path.join(BASE_LOG_DIR, LOG_DATE, FOLD_ID)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(LOG_DIR, f'gold_ai_v{__version__}_qa.log')
# minimum rows required for trade log validation (override via TRADE_LOG_MIN_ROWS)
DEFAULT_TRADE_LOG_MIN_ROWS = int(os.getenv("TRADE_LOG_MIN_ROWS", 10))

# ตั้งค่า Logger กลางเพื่อให้โมดูลอื่น ๆ ใช้งานร่วมกัน
# [Patch v5.5.6] Force COMPACT_LOG when running under pytest
if os.environ.get('PYTEST_CURRENT_TEST'):
    os.environ['COMPACT_LOG'] = '1'
# [Patch v5.4.1] รองรับโหมด COMPACT_LOG เพื่อลดข้อความที่แสดงบนหน้าจอ
_compact_log = os.environ.get('COMPACT_LOG', '0') == '1'
_log_level_name = 'WARNING' if _compact_log else os.environ.get('LOG_LEVEL', 'INFO').upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)
logger.setLevel(_log_level)
formatter = logging.Formatter(
    '[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(lineno)d] - %(message)s'
)
fh = logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8')
fh.setFormatter(formatter)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
for h in logger.handlers:
    try: h.close()
    except Exception: pass
logger.handlers.clear()
logger.addHandler(fh)
logger.addHandler(sh)
logger.propagate = True  # [Patch v5.3.8] Propagate to root for testing
atexit.register(logging.shutdown)
root_logger = logging.getLogger()
root_logger.setLevel(_log_level)
logger.info(f"--- (Start) Gold AI v{__version__} ---")
logger.info("--- กำลังโหลดไลบรารีและตรวจสอบ Dependencies ---")

# --- Library Installation & Checks ---
# Helper function to check and log library version
# [Patch v5.0.2] Exclude log_library_version from coverage
def log_library_version(library_name, library_object=None, version=None):  # pragma: no cover
    """Logs the version of the imported library."""
    # [Patch v5.1.0] ยืนยันว่าฟังก์ชันใช้ตัวแปร logger ที่นำเข้าไว้ด้านบน
    try:
        version = version or getattr(library_object, '__version__', 'N/A')
        logger.info(f"   (Info) Using {library_name} version: {version}")
    except Exception as e:
        logger.warning(f"   (Warning) Could not retrieve {library_name} version: {e}")

# Log versions of core libraries
log_library_version("Pandas", pd)
log_library_version("NumPy", np)
try:
    import sklearn
    log_library_version("Scikit-learn", sklearn)
except ImportError:
    logging.warning("   (Warning) Scikit-learn not imported yet.")


# tqdm library
try:
    from tqdm.notebook import tqdm
    logging.debug("tqdm library already installed.")
except ImportError:
    if AUTO_INSTALL_LIBS:
        logging.info("   กำลังติดตั้งไลบรารี tqdm...")
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", "tqdm", "-q"],
                check=True, capture_output=True, text=True,
            )
            logging.debug(f"   ผลการติดตั้ง tqdm: ...{process.stdout[-200:]}")
            from tqdm.notebook import tqdm
            logging.info("   (Success) ติดตั้ง tqdm สำเร็จ.")
        except Exception as e_install:
            logging.error(f"   (Error) ไม่สามารถติดตั้ง tqdm: {e_install}", exc_info=True)
            tqdm = None
    else:
        logging.error("ไลบรารี 'tqdm' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False")
        tqdm = None

# [Patch v4.8.12] Ensure TA library is installed once then record version
TA_VERSION = "N/A"

# [Patch v5.0.2] Exclude TA auto-install from coverage
def _ensure_ta_installed():  # pragma: no cover
    """Ensure `ta` library is available and record its version."""
    global ta, TA_VERSION
    try:
        import ta  # noqa: F401
    except ImportError:
        if AUTO_INSTALL_LIBS:
            logging.info("(Info) ไลบรารี 'ta' ไม่พบ กำลังติดตั้งอัตโนมัติ...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ta"])
                importlib.invalidate_caches()
                import ta as _ta
            except Exception as e_install:
                logging.warning(f"(Warning) ติดตั้งไลบรารี ta ไม่สำเร็จ: {e_install}")
                TA_VERSION = None
                return
            ta = _ta
        else:
            logging.warning(
                "Library 'ta' not found. Please run 'pip install -r requirements.txt' to install all dependencies."
            )
            TA_VERSION = None
            return
    TA_VERSION = getattr(ta, "__version__", None)
    if TA_VERSION is None:
        try:
            from importlib.metadata import version as _v
            TA_VERSION = _v("ta")
        except Exception:
            TA_VERSION = "N/A"
    globals()["ta"] = ta
    # [Patch v6.9.13] Log TA version using resolved TA_VERSION
    log_library_version("TA", ta, version=TA_VERSION)


_ensure_ta_installed()

# Optuna library
# pragma: no cover
try:
    import optuna
    logging.debug("Optuna library already installed.")
    log_library_version("Optuna", optuna)
    # Consider setting verbosity later if needed
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    if AUTO_INSTALL_LIBS:
        logging.info("   กำลังติดตั้งไลบรารี optuna...")
        try:
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", "optuna", "-q"],
                check=True, capture_output=True, text=True,
            )
            logging.debug(f"   ผลการติดตั้ง optuna: ...{process.stdout[-200:]}")
            import optuna
            logging.info("   (Success) ติดตั้ง optuna สำเร็จ.")
            log_library_version("Optuna", optuna)
            # optuna.logging.set_verbosity(optuna.logging.WARNING)
        except Exception as e_install:
            logging.error(
                f"   (Error) ไม่สามารถติดตั้ง optuna: {e_install}. Hyperparameter Optimization จะไม่ทำงาน.",
                exc_info=True,
            )
            optuna = None
    else:
        logging.warning(
            "ไลบรารี 'optuna' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False -- ข้ามการปรับแต่ง"
        )
        optuna = None
# pragma: cover

# XGBoost (Removed in v3.6.6)
XGBClassifier = None
logging.debug("XGBoost is not used in this version.")

# CatBoost library
# pragma: no cover
try:
    import catboost
    from catboost import CatBoostClassifier, Pool
    logging.info(f"   (Info) ตรวจสอบเวอร์ชัน CatBoost: {catboost.__version__}")
    try:
        from catboost.utils import get_gpu_device_count
        gpu_count = get_gpu_device_count()
        logging.info(f"   (Info) ตรวจสอบจำนวน GPU สำหรับ CatBoost: {gpu_count}")
    except Exception as e_cb_gpu_check:
        logging.warning(f"   (Warning) ไม่สามารถตรวจสอบจำนวน GPU ของ CatBoost: {e_cb_gpu_check}")
except ImportError:
    if AUTO_INSTALL_LIBS:
        logging.info("   กำลังติดตั้งไลบรารี catboost...")
        try:
            install_command = [sys.executable, "-m", "pip", "install", "catboost", "-q"]
            process = subprocess.run(
                install_command,
                check=True, capture_output=True, text=True,
            )
            logging.debug(f"   ผลการติดตั้ง catboost: ...{process.stdout[-200:]}")
            import catboost
            from catboost import CatBoostClassifier, Pool
            logging.info(
                f"   (Success) ติดตั้ง catboost สำเร็จ (เวอร์ชัน: {catboost.__version__})."
            )
            try:
                from catboost.utils import get_gpu_device_count
                gpu_count_post = get_gpu_device_count()
                logging.info(
                    f"   (Info) ตรวจสอบจำนวน GPU สำหรับ CatBoost (หลังติดตั้ง): {gpu_count_post}"
                )
            except Exception as e_cb_gpu_check_post:
                logging.warning(
                    f"   (Warning) ไม่สามารถตรวจสอบจำนวน GPU ของ CatBoost (หลังติดตั้ง): {e_cb_gpu_check_post}"
                )
        except Exception as e_cat_install:
            logging.error(
                f"   (Error) ไม่สามารถติดตั้ง catboost: {e_cat_install}. CatBoost models และ SHAP อาจไม่ทำงาน.",
                exc_info=True,
            )
            CatBoostClassifier = None
            Pool = None
            catboost = None
    else:
        logging.warning(
            "ไลบรารี 'catboost' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False -- ข้ามขั้นตอน CatBoost"
        )
        CatBoostClassifier = None
        Pool = None
        catboost = None
# pragma: cover

# psutil library
# pragma: no cover
try:
    import psutil
    logging.debug("psutil library already installed.")
    log_library_version("psutil", psutil)
except ImportError:
    if AUTO_INSTALL_LIBS:
        logging.info("   กำลังติดตั้ง psutil สำหรับตรวจสอบ RAM...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "-q"], check=True)
            import psutil
            logging.info("   (Success) ติดตั้ง psutil สำเร็จ.")
            log_library_version("psutil", psutil)
        except Exception as e_install:
            logging.error(f"   (Error) ไม่สามารถติดตั้ง psutil: {e_install}", exc_info=True)
            psutil = None
    else:
        logging.error("ไลบรารี 'psutil' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False")
        psutil = None
# pragma: cover

# SHAP library
# pragma: no cover
SHAP_INSTALLED = False
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_INSTALLED = True
    SHAP_AVAILABLE = True
    logging.debug("shap library already installed.")
    log_library_version("SHAP", shap)
except ImportError:
    if AUTO_INSTALL_LIBS:
        logging.info("   กำลังติดตั้งไลบรารี shap...")
        try:
            logging.info("      (การติดตั้ง SHAP อาจใช้เวลาสักครู่...)")
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", "shap", "-q"],
                check=True, capture_output=True, text=True,
            )
            logging.debug(f"   ผลการติดตั้ง shap: ...{process.stdout[-200:]}")
            import shap
            SHAP_INSTALLED = True
            SHAP_AVAILABLE = True
            logging.info("   (Success) ติดตั้ง shap สำเร็จ.")
            log_library_version("SHAP", shap)
        except Exception as e_shap_install:
            logging.error(
                f"   (Error) ไม่สามารถติดตั้ง shap: {e_shap_install}. การวิเคราะห์ SHAP จะถูกข้ามไป.",
                exc_info=True,
            )
        shap = None
    else:
        logging.warning(
            "ไลบรารี 'shap' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False -- ข้ามการคำนวณ SHAP"
        )
        shap = None
# pragma: cover


def install_shap():
    """Install the shap library if not already available."""
    global SHAP_INSTALLED, SHAP_AVAILABLE, shap
    if SHAP_INSTALLED:
        return
    logging.info("   กำลังติดตั้งไลบรารี shap...")
    try:
        logging.info("      (การติดตั้ง SHAP อาจใช้เวลาสักครู่...)")
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", "shap", "-q"],
            check=True, capture_output=True, text=True,
        )
        logging.debug(f"   ผลการติดตั้ง shap: ...{process.stdout[-200:]}")
        import shap
        SHAP_INSTALLED = True
        SHAP_AVAILABLE = True
        logging.info("   (Success) ติดตั้ง shap สำเร็จ.")
        log_library_version("SHAP", shap)
    except Exception as e_shap_install:
        logging.error(
            f"   (Error) ไม่สามารถติดตั้ง shap: {e_shap_install}. การวิเคราะห์ SHAP จะถูกข้ามไป.",
            exc_info=True,
        )
        shap = None

# GPUtil library (Optional for Resource Monitor)
# [Patch v5.10.2] broaden exception handling for GPU imports
# pragma: no cover
try:
    import GPUtil
    logging.debug("GPUtil library already installed.")
except Exception as e_gp_import:
    if AUTO_INSTALL_LIBS:
        logging.info("   กำลังติดตั้ง GPUtil สำหรับตรวจสอบ GPU (Optional)...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "gputil", "-q"], check=True)
            import GPUtil
            logging.info("   (Success) ติดตั้ง GPUtil สำเร็จ.")
        except Exception as e_install:
            logging.warning(
                f"   (Warning) ไม่สามารถติดตั้ง GPUtil: {e_install}. ฟังก์ชัน show_system_status อาจไม่ทำงาน."
            )
            GPUtil = None
    else:
        logging.warning("ไลบรารี 'GPUtil' ไม่ถูกติดตั้ง หรือไม่สามารถโหลดได้")
        logging.debug(f"GPUtil import error: {e_gp_import}")
        GPUtil = None
# pragma: cover

# --- Colab/Drive Setup ---
def is_colab():
    """Return True if running within Google Colab."""  # [Patch v5.4.9]
    # [Patch] Require an interactive kernel to avoid mount errors when running
    # scripts externally. Environment variables alone are not sufficient.
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        try:
            import google.colab  # noqa: F401
            ip = get_ipython()
            if ip and getattr(ip, "kernel", None):
                return True
        except Exception:
            return False
    try:
        ip = get_ipython()
        return (
            bool(ip)
            and getattr(ip, "kernel", None) is not None
            and "google.colab" in str(ip.__class__)
        )
    except Exception:
        return False

FILE_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILE_BASE_OVERRIDE = os.getenv("FILE_BASE_OVERRIDE")
if FILE_BASE_OVERRIDE and os.path.isdir(FILE_BASE_OVERRIDE):
    FILE_BASE = FILE_BASE_OVERRIDE
elif is_colab():
    from google.colab import drive
    logging.info("(Info) รันบน Google Colab – กำลัง mount Google Drive...")
    try:
        drive.mount('/content/drive', force_remount=True)
        logging.info("(Success) Mount Google Drive สำเร็จ")
        # ใช้โฟลเดอร์ปัจจุบันเป็นฐานข้อมูลเพื่อให้ทำงานได้ทุกที่
        FILE_BASE = os.getcwd()
    except Exception as e_drive:
        logging.warning(
            f"(Warning) ล้มเหลวในการ mount Drive: {e_drive} -- ดำเนินการต่อโดยใช้ Local Path แทน"
        )
        if FILE_BASE_OVERRIDE and os.path.isdir(FILE_BASE_OVERRIDE):
            FILE_BASE = FILE_BASE_OVERRIDE
else:
    logging.info(
        "(Info) ไม่ใช่ Colab – ใช้เส้นทางในเครื่องสำหรับจัดเก็บ log และข้อมูล"
    )

DEFAULT_CSV_PATH_M1 = os.path.join(FILE_BASE, "XAUUSD_M1.csv")
DEFAULT_CSV_PATH_M15 = os.path.join(FILE_BASE, "XAUUSD_M15.csv")
DEFAULT_LOG_DIR = BASE_LOG_DIR

# [Patch v6.2.1] provide base data directory and default symbol/timeframe
DATA_DIR = Path(os.getenv("DATA_DIR", FILE_BASE))
SYMBOL = os.getenv("SYMBOL", "XAUUSD")
TIMEFRAME = os.getenv("TIMEFRAME", "M1")


# --- GPU Acceleration Setup (Optional) ---
# pragma: no cover
USE_GPU_ACCELERATION = os.getenv("USE_GPU_ACCELERATION", "True").lower() in ("true", "1", "yes")
cudf = None; cuml = None; cuStandardScaler = None; pynvml = None; nvml_handle = None
logging.info("   (Checking) กำลังตรวจสอบความพร้อมใช้งาน GPU...")
try:
    import torch
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"   (Success) พบ GPU: {gpu_name}")
            try:  # [Patch v5.10.2] handle pynvml import errors
                import pynvml
                pynvml.nvmlInit()
                nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logging.info("   (Success) เริ่มต้น pynvml สำหรับการตรวจสอบ GPU สำเร็จ.")
            except Exception as e_nvml:
                logging.warning(
                    "   ไลบรารี 'pynvml' ไม่ถูกติดตั้ง หรือไม่สามารถโหลดได้ -- ข้ามการตรวจสอบ GPU"
                )
                logging.debug(f"pynvml error: {e_nvml}")
                pynvml = None
                if nvml_handle:
                    try:
                        pynvml.nvmlShutdown()
                    except Exception:
                        pass
                    nvml_handle = None
        else:
            logging.info("   (Info) PyTorch ไม่พบ GPU หรือ CUDA ไม่พร้อม. การเร่งความเร็วด้วย GPU จะถูกปิด.")
            USE_GPU_ACCELERATION = False
    except Exception as e_cuda:
        logging.warning(
            "   (Warning) ไม่สามารถใช้งาน CUDA ได้ -- ปิด GPU Acceleration."
        )
        logging.debug(f"CUDA check error: {e_cuda}")
        USE_GPU_ACCELERATION = False
# [Patch v6.2.3] Handle PyTorch import errors concisely
except Exception as e_torch_import:
    logging.warning(f"(Warning) GPU acceleration disabled due to import error: {e_torch_import}")
    USE_GPU_ACCELERATION = False
logging.info(f"(Info) GPU acceleration status: {USE_GPU_ACCELERATION}")
# pragma: cover

# --- GPU/RAM Utilization Helper Function ---
# [Patch v5.0.2] Exclude GPU utilization logger from coverage
def print_gpu_utilization(context=""):  # pragma: no cover
    """Logs GPU and RAM utilization if available."""
    gpu_util_str = "N/A"; gpu_mem_str = "N/A"; ram_str = "N/A"
    global nvml_handle, pynvml

    if USE_GPU_ACCELERATION and pynvml and nvml_handle:
        try:
            info = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            gpu_util_str = f"{info.gpu}%"
            gpu_mem_str = f"{info.memory}% ({mem_info.used // 1024**2}MB / {mem_info.total // 1024**2}MB)"
        except pynvml.NVMLError as e_gpu_mon:
            gpu_util_str = "NVML Err"
            gpu_mem_str = f"NVML Err: {e_gpu_mon}"
            logging.warning(f"NVML Error during GPU monitoring: {e_gpu_mon}. Disabling pynvml monitoring.")
            if pynvml:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            nvml_handle = None
            pynvml = None
        except Exception as e_gpu_mon_other:
            gpu_util_str = "Err"
            gpu_mem_str = f"Err: {e_gpu_mon_other}"
            logging.warning(f"Unexpected error retrieving GPU stats: {e_gpu_mon_other}.")
            if pynvml:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            nvml_handle = None
            pynvml = None
    elif USE_GPU_ACCELERATION and not pynvml:
        gpu_util_str = "pynvml N/A"; gpu_mem_str = "pynvml N/A"
    elif not USE_GPU_ACCELERATION:
        gpu_util_str = "Disabled"; gpu_mem_str = "Disabled"

    if psutil:
        try:
            mem = psutil.virtual_memory()
            ram_str = f"{mem.percent}% ({mem.used // 1024**2}MB / {mem.total // 1024**2}MB)"
        except Exception as e_mem:
            ram_str = "N/A"
            logging.warning(f"Unable to retrieve RAM stats: {e_mem}.")
    else:
        ram_str = "psutil not installed"

    logging.info(f"[{context}] GPU Util: {gpu_util_str} | Mem: {gpu_mem_str} | RAM: {ram_str}")

# --- [Optional] System Status Monitor using GPUtil ---
# [Patch v5.0.2] Exclude system status monitor from coverage
def show_system_status(context=""):  # pragma: no cover
    """Logs system resource usage (RAM and GPU using GPUtil)."""
    ram_str = "N/A"; gpu_str_list = ["N/A"]
    if psutil:
        try:
            ram_info = psutil.virtual_memory()
            ram_str = f"{ram_info.percent:.1f}% ({ram_info.used // 1024**2}MB / {ram_info.total // 1024**2}MB)"
        except Exception as e_ram_mon:
            ram_str = f"Error: {e_ram_mon}"
            logging.error(f"Error getting RAM info (GPUtil func): {e_ram_mon}", exc_info=True)
    else:
        ram_str = "psutil N/A"

    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_str_list = []
                for gpu in gpus:
                    gpu_str_list.append(f"GPU {gpu.id} {gpu.name} | Load: {gpu.load*100:.1f}% | Mem: {gpu.memoryUtil*100:.1f}% ({gpu.memoryUsed:.0f}MB/{gpu.memoryTotal:.0f}MB)")
            else:
                gpu_str_list = ["No GPU found by GPUtil"]
        except Exception as e_gpu_mon:
            gpu_str_list = [f"GPUtil Error: {e_gpu_mon}"]
            logging.error(f"Error getting GPU info via GPUtil: {e_gpu_mon}", exc_info=True)
    else:
        gpu_str_list = ["GPUtil N/A"]

    logging.info(f"[{context}] RAM: {ram_str} | {' | '.join(gpu_str_list)}")

# [Patch v6.3.1] ตรวจสอบจำนวน GPU สำหรับ PyTorch และจับข้อผิดพลาด MKL
def check_gpu_availability():
    """Return the number of GPUs available or 0 if unavailable."""
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        logger.info(f"(Info) ตรวจสอบจำนวน GPU สำหรับ PyTorch: {gpu_count}")
    except Exception as e:
        logger.warning(
            f"(Warning) ไม่สามารถโหลด PyTorch หรือ MKL module: {e} — ใช้ CPU เท่านั้น"
        )
        gpu_count = 0
    return gpu_count


# === Global Settings and Warnings ===
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")
warnings.filterwarnings("ignore", message="Using CPU via PyArrow to expand CUDF columns")
pd.options.mode.chained_assignment = None
logging.debug("Global warnings filtered and pandas options set.")

# ==============================================================================
# === CONFIGURATION (v4.8.4) ===
# ==============================================================================
logging.info("Loading Global Configuration Settings...")
OUTPUT_BASE_DIR = DEFAULT_LOG_DIR
OUTPUT_DIR_NAME = f"outputgpt_v{__version__}"
DATA_FILE_PATH_M15 = DEFAULT_CSV_PATH_M15
DATA_FILE_PATH_M1 = DEFAULT_CSV_PATH_M1
DEFAULT_TRADE_LOG_MIN_ROWS = 10  # Minimum rows required in trade log
TRAIN_META_MODEL_BEFORE_RUN = True
DEFAULT_MODEL_TO_LINK = "catboost"
META_CLASSIFIER_PATH = "meta_classifier.pkl"
SPIKE_MODEL_PATH = "meta_classifier_spike.pkl"
CLUSTER_MODEL_PATH = "meta_classifier_cluster.pkl"
pattern_label_map = {
    "Breakout": 0, "Reversal": 1, "InsideBar": 2, "StrongTrend": 3,
    "Choppy": 4, "Normal": 5, "N/A": 5,
}
logging.debug(f"Pattern Label Map: {pattern_label_map}")

# Default timezone used when localizing naive datetime values
DEFAULT_NAIVE_TZ = "UTC"

# Default data types for CSV loading to reduce memory usage
DEFAULT_DTYPE_MAP = {
    "Open": "float32",
    "High": "float32",
    "Low": "float32",
    "Close": "float32",
    "Volume": "float32",
}

# --- Multi-Fund & IB Config ---
MULTI_FUND_MODE = True
FUND_PROFILES = {
    "SAFE": {"risk": 0.005, "mm_mode": "conservative"},
    "NORMAL": {"risk": 0.01, "mm_mode": "balanced"},
    "SPIKE": {"risk": 0.02, "mm_mode": "spike_only"},
    "AGGRESSIVE": {"risk": 0.03, "mm_mode": "high_freq"},
    "MIRROR": {"risk": 0.01, "mm_mode": "mirror"},
}
DEFAULT_FUND_NAME = "NORMAL"
IB_COMMISSION_PER_LOT = 7.0

from dataclasses import dataclass

@dataclass
class DefaultConfig:
    OUTPUT_DIR: str = str(OUTPUT_DIR)
    DATA_FILE_PATH_M1: str = DEFAULT_CSV_PATH_M1
    DATA_FILE_PATH_M15: str = DEFAULT_CSV_PATH_M15
    DEFAULT_RISK_PER_TRADE: float = FUND_PROFILES.get(DEFAULT_FUND_NAME, {}).get("risk", 0.01)
    # [Patch v5.9.3] Default hyperparameters for CatBoost training
    LEARNING_RATE: float = 0.01
    DEPTH: int = 6
    L2_LEAF_REG: float | None = None
logging.info(f"Multi-Fund Mode: {MULTI_FUND_MODE}")
if MULTI_FUND_MODE:
    logging.info(f"Fund Profiles: {list(FUND_PROFILES.keys())}")
else:
    logging.info(f"Default Fund Profile: {DEFAULT_FUND_NAME}")
logging.info(f"IB Commission per Lot: ${IB_COMMISSION_PER_LOT:.2f}")

global df_m15_dt
df_m15_dt = None
logging.debug("Placeholder df_m15_dt initialized to None.")

logging.info("Part 1: Setup & Configuration Complete.")
# === END OF PART 1/12 ===
# === START OF PART 2/12 ===

# ==============================================================================
# === PART 2: Core Parameters & Strategy Settings (v4.8.1) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Added logging info, minor comment updates >>>
# <<< Includes fix from v4.7.8_p2_fix: Added MAX_DRAWDOWN_THRESHOLD definition >>>
# <<< MODIFIED v4.8.1: Version bump in comments, parameters remain as per previous version >>>
import logging # Ensure logging is available if this part is run independently

logging.info("Loading Core Parameters & Strategy Settings...")

# --- Backtesting Parameters ---
logging.debug("Setting Backtesting Parameters...")
N_WALK_FORWARD_SPLITS = 5       # Number of folds for final backtest (increased to 5)
INITIAL_CAPITAL = 100.0         # Starting capital for simulation
POINT_VALUE = 0.1               # Value per point for 0.01 lot size
MAX_CONCURRENT_ORDERS = 7       # [Patch v5.3.5] Max concurrent orders per side (BUY/SELL)
MAX_HOLDING_BARS = 24           # Max bars an order can be held
COMMISSION_PER_001_LOT = 0.10   # Commission per 0.01 lot (USD)
SPREAD_POINTS = 2.0             # Fixed spread in points
MIN_SLIPPAGE_POINTS = -5.0      # Minimum slippage in points (negative means better price)
MAX_SLIPPAGE_POINTS = -1.0      # Maximum slippage in points (negative means better price)
OMS_MARGIN_PIPS = 20.0          # [Patch v5.5.8] Minimum SL distance from entry
OMS_MAX_DISTANCE_PIPS = 1000.0  # [Patch v5.5.8] Max allowed SL/TP distance

# --- Entry/Exit Logic Parameters ---
logging.debug("Setting Entry/Exit Logic Parameters...")
# [Patch v6.9.16] Load default from settings with env override
MIN_SIGNAL_SCORE_ENTRY = get_env_float(
    "MIN_SIGNAL_SCORE_ENTRY", SETTINGS.min_signal_score_entry
)
# [Patch v5.3.9] Adaptive threshold settings
ADAPTIVE_SIGNAL_SCORE_WINDOW = 1000   # Bars used for quantile calculation
ADAPTIVE_SIGNAL_SCORE_QUANTILE = 0.4  # [Patch v5.7.1] Lower quantile (40th)
MIN_SIGNAL_SCORE_ENTRY_MIN = 0.3      # Clamp lower bound
MIN_SIGNAL_SCORE_ENTRY_MAX = 3.0      # Clamp upper bound
MIN_SIGNAL_SCORE_ENTRY = max(MIN_SIGNAL_SCORE_ENTRY_MIN, min(MIN_SIGNAL_SCORE_ENTRY_MAX, MIN_SIGNAL_SCORE_ENTRY))
USE_ADAPTIVE_SIGNAL_SCORE = True
BASE_TP_MULTIPLIER = 1.8        # Base R-multiple for TP2 (before dynamic adjustment)
BASE_BE_SL_R_THRESHOLD = 1.0    # Base R-multiple threshold to move SL to Breakeven
ADAPTIVE_TSL_START_ATR_MULT = 1.5 # ATR multiplier from entry price to start Trailing Stop Loss
ADAPTIVE_TSL_DEFAULT_STEP_R = 0.5 # Default TSL step size (in R units) in normal volatility
ADAPTIVE_TSL_HIGH_VOL_RATIO = 1.8 # ATR ratio (current/avg) considered high volatility
ADAPTIVE_TSL_HIGH_VOL_STEP_R = 1.0 # TSL step size (R units) during high volatility
ADAPTIVE_TSL_LOW_VOL_RATIO = 0.75 # ATR ratio considered low volatility
ADAPTIVE_TSL_LOW_VOL_STEP_R = 0.3  # TSL step size (R units) during low volatility
MAX_WICK_RATIO_ENTRY = 0.75     # Maximum candle wick ratio allowed for entry (prevents spikes)
DYNAMIC_BE_ATR_THRESHOLD_HIGH = 1.2 # ATR ratio threshold to adjust BE R-threshold
DYNAMIC_BE_R_ADJUST_HIGH = 0.2  # R value added to BE threshold during high volatility
M1_ENTRY_CANDLE_RATIO_THRESH = 0.85 # (Not directly used in current logic, kept for potential future use)
M1_ENTRY_MACD_HIST_THRESH = -0.1  # (Not directly used in current logic, kept for potential future use)
FAST_MA_PERIOD = 10  # [Patch v6.5.9] Fallback MA fast period
SLOW_MA_PERIOD = 30  # [Patch v6.5.9] Fallback MA slow period
M15_TREND_EMA_FAST = 50         # Fast EMA period for M15 Trend Filter
M15_TREND_EMA_SLOW = 200        # Slow EMA period for M15 Trend Filter
M15_TREND_RSI_PERIOD = 14       # RSI period for M15 Trend Filter
# ↓↑ เพิ่มพื้นที่ให้สีเขียว/แดงบน M15 มากขึ้น เพื่อให้กรองเทรนด์ไม่เข้มงวดจนเกินไป
M15_TREND_RSI_UP = 60           # [Patch v5.6.4] Relaxed M15 trend zone thresholds
M15_TREND_RSI_DOWN = 40         # [Patch v5.6.4] Relaxed M15 trend zone thresholds

session_env = os.getenv("SESSION_TIMES_UTC")
try:
    SESSION_TIMES_UTC = json.loads(session_env) if session_env else {"Asia": (22, 8), "London": (7, 16), "NY": (13, 21)}
except Exception:
    logging.warning("(Warning) SESSION_TIMES_UTC env var invalid. Using default.")
    SESSION_TIMES_UTC = {"Asia": (22, 8), "London": (7, 16), "NY": (13, 21)}
logging.debug(f"Session Times (UTC): {SESSION_TIMES_UTC}")

# --- Signal Toggle Configuration ---
# (Deprecated: moved to module top-level)

# --- Fold-Specific Configuration ---
# Allows overriding parameters for specific walk-forward folds
logging.debug("Setting Fold-Specific Configuration...")
ENTRY_CONFIG_PER_FOLD = {
    # Fold Index: {Config Dictionary}
    0: {"sl_multiplier": 2.8, "gain_z_thresh": 0.3, "cooldown_sec": 0, "min_signal_score": MIN_SIGNAL_SCORE_ENTRY, },
    1: {"sl_multiplier": 2.0, "gain_z_thresh": 0.3, "cooldown_sec": 0, "min_signal_score": MIN_SIGNAL_SCORE_ENTRY, },
    2: {"sl_multiplier": 2.0, "gain_z_thresh": 0.3, "cooldown_sec": 0, "min_signal_score": MIN_SIGNAL_SCORE_ENTRY, },
    3: {"sl_multiplier": 2.0, "gain_z_thresh": 0.3, "cooldown_sec": 0, "min_signal_score": MIN_SIGNAL_SCORE_ENTRY, },
    4: {"sl_multiplier": 2.0, "gain_z_thresh": 0.3, "cooldown_sec": 0, "min_signal_score": MIN_SIGNAL_SCORE_ENTRY, },
}
logging.debug(f"Entry Config Per Fold (First Fold Example): {ENTRY_CONFIG_PER_FOLD.get(0, {})}")

# --- Order Management System (OMS) Configuration ---
logging.debug("Setting Order Management System (OMS) Configuration...")
ENABLE_PARTIAL_TP = True        # Enable/disable partial take profit logic
PARTIAL_TP_LEVELS = [           # Define partial TP levels
    {"r_multiple": 0.25, "close_pct": 0.5},  # Close 50% at 0.5 ATR
    {"r_multiple": 0.5, "close_pct": 0.5},   # Close remaining at 1 ATR
]
PARTIAL_TP_MOVE_SL_TO_ENTRY = True # Move SL to entry after first partial TP?
OMS_DEFAULT = True  # Default OMS state when not overridden
OMS_ENABLED = OMS_DEFAULT  # Global switch to enable/disable OMS
PAPER_MODE = False  # When True, bypass OMS block checks for paper trading
ENABLE_KILL_SWITCH = True       # Enable/disable kill switch mechanism
KILL_SWITCH_MAX_DD_THRESHOLD = 0.30 # ↑ Relax kill switch ให้เลิก block ช้าลง (30% drawdown)
KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 5 # [Patch] Lower threshold for earlier soft cooldown
MAX_DRAWDOWN_THRESHOLD = 0.10   # [Patch v6.1.6] Tighter drawdown block threshold
logging.info(f"OMS Enabled: {OMS_ENABLED}")
logging.info(f"Kill Switch Enabled: {ENABLE_KILL_SWITCH} (DD > {KILL_SWITCH_MAX_DD_THRESHOLD*100:.0f}%, Losses > {KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD})")
logging.info(f"Max Drawdown Threshold (Block New Orders): {MAX_DRAWDOWN_THRESHOLD*100:.0f}%")

# --- Spike Guard & Recovery Mode Configuration ---
logging.debug("Setting Spike Guard & Recovery Mode Configuration...")
ENABLE_SPIKE_GUARD = True       # Enable/disable spike guard filter (mainly London session)
ENABLE_SOFT_COOLDOWN = True     # Enable/disable soft cooldown logic
POST_TRADE_COOLDOWN_BARS = 2  # Bars after closing trade before allowing new entry
RECOVERY_MODE_CONSECUTIVE_LOSSES = 4 # Consecutive losses to enter recovery mode
RECOVERY_MODE_LOT_MULTIPLIER = 0.3 # [Patch v6.1.6] Reduce lot size in recovery mode
logging.info(f"Spike Guard Enabled: {ENABLE_SPIKE_GUARD}")
logging.info(f"Post-Trade Cooldown: {POST_TRADE_COOLDOWN_BARS} bars")
logging.info(f"Soft Cooldown Enabled: {ENABLE_SOFT_COOLDOWN}")
logging.info(f"Recovery Mode Enabled: Losses >= {RECOVERY_MODE_CONSECUTIVE_LOSSES}, Lot Multiplier: {RECOVERY_MODE_LOT_MULTIPLIER}")

# --- Re-Entry Configuration ---
logging.debug("Setting Re-Entry Configuration...")
USE_REENTRY = True              # Enable/disable re-entry logic
REENTRY_COOLDOWN_BARS = 1       # Cooldown (in bars) after TP before allowing re-entry
REENTRY_MIN_PROBA_THRESH = 0.40 # ↓ ลดเงื่อนไขให้ยอมรับ ML probability ที่ต่ำกว่า (เพื่อ Re-entry บ่อยขึ้น)
logging.info(f"Re-Entry Enabled: {USE_REENTRY} (Cooldown: {REENTRY_COOLDOWN_BARS} bars, Threshold: {REENTRY_MIN_PROBA_THRESH})")

# --- Meta Filter Configuration ---
logging.debug("Setting Meta Filter Configuration...")
META_FILTER_THRESHOLD = get_env_float(
    "META_FILTER_THRESHOLD", SETTINGS.meta_filter_threshold
)
# ↓ ลดค่า Meta Filter เพื่อให้ผ่านโลจิก Meta-Model บ่อยขึ้น
META_FILTER_RELAXED_THRESHOLD = get_env_float(
    "META_FILTER_RELAXED_THRESHOLD", SETTINGS.meta_filter_relaxed_threshold
)
META_FILTER_RELAX_BLOCKS = int(
    get_env_float("META_FILTER_RELAX_BLOCKS", SETTINGS.meta_filter_relax_blocks)
)
logging.info(
    f"Meta Filter Threshold: {META_FILTER_THRESHOLD} (Relaxed: {META_FILTER_RELAXED_THRESHOLD}, Blocks: {META_FILTER_RELAX_BLOCKS})"
)

# --- Forced Entry Configuration ---
logging.debug("Setting Forced Entry Configuration...")
ENABLE_FORCED_ENTRY = True      # Enable/disable forced entry logic
logging.info(f"   *** (Config Info) Forced Entry is ENABLED: {ENABLE_FORCED_ENTRY} ***")
FORCED_ENTRY_BAR_THRESHOLD = 100 # Bars without trades before considering forced entry
FORCED_ENTRY_MIN_SIGNAL_SCORE = 0.5 # Minimum signal score for forced entry
FORCED_ENTRY_LOOKBACK_PERIOD = 500 # Lookback period (bars) for other FE conditions
USE_MACD_FOR_FORCED_ENTRY = False # (Not used)
USE_GAIN_Z_FOR_FORCED_ENTRY = False # (Not used)
USE_CANDLE_RATIO_FOR_FORCED_ENTRY = False # (Not used)
FORCED_ENTRY_CHECK_MARKET_COND = True # Check market conditions (ATR, GainZ) before FE?
FORCED_ENTRY_MAX_ATR_MULT = 2.5 # Max ATR multiplier allowed for FE
FORCED_ENTRY_MIN_GAIN_Z_ABS = 0.5 # [Patch v5.6.4] Lower Gain_Z requirement for FE
FORCED_ENTRY_ALLOWED_REGIMES = ["Normal", "Breakout", "StrongTrend", "Reversal", "InsideBar", "Choppy"] # [Patch v5.6.4] Allow more patterns for FE
FE_ML_FILTER_THRESHOLD = 0.40   # ML probability threshold for filtering FE (if used)
ADAPTIVE_COOLDOWN_LOSS_MULTIPLIER = 0.2 # (Not directly used in FE, kept for potential future use)
forced_entry_max_consecutive_losses = 2 # Max consecutive FE losses before temporary disable
min_equity_threshold_pct = 0.70 # Min equity % of initial capital before disabling FE
logging.debug(f"Forced Entry Config: Bars>{FORCED_ENTRY_BAR_THRESHOLD}, Score>{FORCED_ENTRY_MIN_SIGNAL_SCORE:.2f}, MktConds:{FORCED_ENTRY_CHECK_MARKET_COND}, MaxLoss:{forced_entry_max_consecutive_losses}")

# --- Money Management (MM) Configuration ---
logging.debug("Setting Money Management (MM) Configuration...")
DEFAULT_RISK_PER_TRADE = 0.01   # Default % risk per trade (for Fixed Risk MM modes)
EQUITY_LOT_REDUCTION_THRESHOLD_PCT = 0.90 # Equity % vs peak before reducing lot (not used in current logic)
MAX_LOT_SIZE = 5.0              # Maximum allowed lot size
MIN_LOT_SIZE = 0.01             # Minimum allowed lot size
logging.info(f"Default Risk Per Trade: {DEFAULT_RISK_PER_TRADE*100:.2f}%")
logging.info(f"Lot Size Limits: Min={MIN_LOT_SIZE}, Max={MAX_LOT_SIZE}")

# --- Feature Engineering Parameters ---
logging.debug("Setting Feature Engineering Parameters...")
TIMEFRAME_MINUTES_M15 = 15
TIMEFRAME_MINUTES_M1 = 1
ROLLING_Z_WINDOW_M1 = 300       # Window for M1 Gain Rolling Z-Score
ATR_ROLLING_AVG_PERIOD = 50     # Window for M1 ATR Rolling Average
# ↓ ทดลองลด Z-Score threshold จาก 2.0 → 1.5 เพื่อให้เกิดสัญญาณ Breakout บ่อยขึ้น
PATTERN_BREAKOUT_Z_THRESH = 1.5 # ↓ ลดสองเท่า เพื่อกรองสัญญาณ Breakout เบาลง
PATTERN_REVERSAL_BODY_RATIO = 0.5 # Current/Previous body ratio for 'Reversal' pattern
PATTERN_STRONG_TREND_Z_THRESH = 0.8 # ↓ ลดเพื่อกรองแนวโน้มแรงเบาลงเล็กน้อย
PATTERN_CHOPPY_CANDLE_RATIO = 0.3 # Min candle ratio for 'Choppy' pattern
PATTERN_CHOPPY_WICK_RATIO = 0.6 # Max wick ratio for 'Choppy' pattern

# --- Drift & Data Quality Configuration ---

logging.debug("Setting Drift & Data Quality Configuration...")

# [Patch v5.5.4] Allow override via environment variable
DRIFT_WASSERSTEIN_THRESHOLD = get_env_float("DRIFT_WASSERSTEIN_THRESHOLD", 0.1)
DRIFT_TTEST_ALPHA = 0.05        # Alpha level for T-test drift detection
SIGNIFICANCE_LEVEL = 0.05       # (Not directly used, kept for potential analysis)
M1_FEATURES_FOR_DRIFT = []      # Will be populated in clean_m1_data (Part 5)
MAX_NAT_RATIO_THRESHOLD = 0.05  # Max allowed NaT ratio after datetime parsing

# Drift override thresholds
RSI_DRIFT_OVERRIDE_THRESHOLD = 0.65  # Threshold to ignore RSI scoring when drift is high
ATR_DRIFT_OVERRIDE_THRESHOLD = 0.25  # Threshold to enable gain-based exit on high ATR drift

# --- Dynamic Adjustment Configuration ---
logging.debug("Setting Dynamic Adjustment Configuration...")
DYNAMIC_GAINZ_DRIFT_THRESHOLD = 0.10 # Wasserstein threshold on Gain_Z to trigger adjustment
DYNAMIC_GAINZ_ADJUSTMENT = 0.1  # Amount to add to Gain_Z entry threshold on high drift
DYNAMIC_RISK_DD_THRESHOLD = 12.0 # Drawdown % to trigger risk reduction (not used)
DYNAMIC_RISK_REDUCTION_FACTOR = 0.7 # Factor to reduce risk by on high DD (not used)
ENABLE_ADAPTIVE_SLTP = False      # Toggle ATR-based SL/TP multipliers
ENABLE_ADAPTIVE_RISK = False      # Toggle dynamic risk allocation
ENABLE_BEST_PARAM_LOGGING = True  # Save best params per fold

logging.info("Part 2: Core Parameters & Strategy Settings Loaded.")
# === END OF PART 2/12 ===

# ------------------------------------------------------------------------------
# ถ้าเรียก get_fund_profile(...) ใน profile_backtest.py จะไม่ Error
# ------------------------------------------------------------------------------
if 'DEFAULT_FUND_NAME' not in globals():
    DEFAULT_FUND_NAME = "DEFAULT"
if 'FUND_PROFILES' not in globals():
    FUND_PROFILES = {DEFAULT_FUND_NAME: {}}
