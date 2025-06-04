# pragma: no cover
# === START OF PART 1/12 ===

# -*- coding: utf-8 -*-
# <<< เพิ่ม Encoding declaration สำหรับอักษรไทย >>>

# ==============================================================================
# === PART 1: Setup & Configuration (v4.8.4) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added basic docstrings/comments >>>
# <<< MODIFIED v4.8.1: Updated versioning for comprehensive fixes based on prompt >>>
# <<< MODIFIED v4.8.4: Updated paths for Colab/VPS compatibility and versioning >>>
import logging
import subprocess
import importlib
import sys
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
import pandas as pd
import numpy as np
# [Patch v5.5.1] Enable auto-installation of libraries
AUTO_INSTALL_LIBS = True  # If False, skip auto-installation of libraries
# อ่านเวอร์ชันจากไฟล์ VERSION
VERSION_FILE = os.path.join(os.path.dirname(__file__), '..', 'VERSION')
with open(VERSION_FILE, 'r', encoding='utf-8') as vf:
    __version__ = vf.read().strip()
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
from src.utils import get_env_float

# --- Logging Configuration ---
# กำหนดค่าพื้นฐานสำหรับการ Logging
# สามารถปรับ level, format, และ filename ได้ตามต้องการ
LOG_FILENAME = f'gold_ai_v{__version__}_qa.log'

# ตั้งค่า Logger กลางเพื่อให้โมดูลอื่น ๆ ใช้งานร่วมกัน
logger = logging.getLogger('NiceGold')
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
def log_library_version(library_name, library_object):  # pragma: no cover
    """Logs the version of the imported library."""
    # [Patch v5.1.0] ยืนยันว่าฟังก์ชันใช้ตัวแปร logger ที่นำเข้าไว้ด้านบน
    try:
        version = getattr(library_object, '__version__', 'N/A')
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
            logging.error("ไลบรารี 'ta' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False")
            TA_VERSION = None
            return
    TA_VERSION = getattr(ta, "__version__", "N/A")
    globals()["ta"] = ta
    log_library_version("TA", ta)


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
# pragma: no cover
try:
    import GPUtil
    logging.debug("GPUtil library already installed.")
except ImportError:
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
        logging.warning("ไลบรารี 'GPUtil' ไม่ถูกติดตั้ง และ AUTO_INSTALL_LIBS=False")
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
        FILE_BASE = "/content/drive/MyDrive/Phiradon168"
    except Exception as e_drive:
        logging.warning(
            f"(Warning) ล้มเหลวในการ mount Drive: {e_drive} -- ดำเนินการต่อโดยใช้ Local Path แทน"
        )
        if FILE_BASE_OVERRIDE and os.path.isdir(FILE_BASE_OVERRIDE):
            FILE_BASE = FILE_BASE_OVERRIDE
else:
    logging.info(
        "(Info) ไม่ใช่ Colab – สมมติรันบน VPS และโฟลเดอร์ Google Drive ถูกซิงก์ไว้เรียบร้อยแล้ว"
    )

DEFAULT_CSV_PATH_M1 = os.path.join(FILE_BASE, "XAUUSD_M1.csv")
DEFAULT_CSV_PATH_M15 = os.path.join(FILE_BASE, "XAUUSD_M15.csv")
DEFAULT_LOG_DIR = os.path.join(FILE_BASE, "logs")


# --- GPU Acceleration Setup (Optional) ---
# pragma: no cover
USE_GPU_ACCELERATION = os.getenv("USE_GPU_ACCELERATION", "True").lower() in ("true", "1", "yes")
cudf = None; cuml = None; cuStandardScaler = None; pynvml = None; nvml_handle = None
logging.info("   (Checking) กำลังตรวจสอบความพร้อมใช้งาน GPU...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"   (Success) พบ GPU: {gpu_name}")
        try:
            import pynvml
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logging.info("   (Success) เริ่มต้น pynvml สำหรับการตรวจสอบ GPU สำเร็จ.")
        except ImportError:
            logging.warning("   (Warning) ไม่พบ pynvml library. GPU monitoring via pynvml disabled.")
            pynvml = None
        except Exception as e_nvml:
            logging.error(f"   (Warning) ข้อผิดพลาด NVML: {e_nvml}.", exc_info=True)
            pynvml = None
            if nvml_handle:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
                nvml_handle = None
    else:
        logging.info("   (Info) PyTorch ไม่พบ GPU. การเร่งความเร็วด้วย GPU จะถูกปิด.")
        USE_GPU_ACCELERATION = False
except ImportError:
    logging.info("   (Info) ไม่พบ PyTorch. การเร่งความเร็วด้วย GPU จะถูกปิด.")
    USE_GPU_ACCELERATION = False
except Exception as e_gpu:
    logging.error(f"   (Error) การตั้งค่า GPU ล้มเหลว: {e_gpu}", exc_info=True)
    if pynvml and nvml_handle:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        nvml_handle = None
    USE_GPU_ACCELERATION = False
logging.info(f"   สถานะการเร่งความเร็วด้วย GPU: {USE_GPU_ACCELERATION}")
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
MIN_SIGNAL_SCORE_ENTRY = 2.0    # Minimum signal score required to open an order
# [Patch v5.3.9] Adaptive threshold settings
ADAPTIVE_SIGNAL_SCORE_WINDOW = 1000   # Bars used for quantile calculation
ADAPTIVE_SIGNAL_SCORE_QUANTILE = 0.7  # Quantile for threshold (e.g., 70th)
MIN_SIGNAL_SCORE_ENTRY_MIN = 0.5      # Clamp lower bound
MIN_SIGNAL_SCORE_ENTRY_MAX = 3.0      # Clamp upper bound
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
M15_TREND_EMA_FAST = 50         # Fast EMA period for M15 Trend Filter
M15_TREND_EMA_SLOW = 200        # Slow EMA period for M15 Trend Filter
M15_TREND_RSI_PERIOD = 14       # RSI period for M15 Trend Filter
M15_TREND_RSI_UP = 52           # RSI threshold for M15 uptrend
M15_TREND_RSI_DOWN = 48         # RSI threshold for M15 downtrend

session_env = os.getenv("SESSION_TIMES_UTC")
try:
    SESSION_TIMES_UTC = json.loads(session_env) if session_env else {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)}
except Exception:
    logging.warning("(Warning) SESSION_TIMES_UTC env var invalid. Using default.")
    SESSION_TIMES_UTC = {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)}
logging.debug(f"Session Times (UTC): {SESSION_TIMES_UTC}")

# --- Signal Toggle Configuration ---
USE_MACD_SIGNALS = True  # Enable MACD conditions in simple signal functions
USE_RSI_SIGNALS = True   # Enable RSI conditions in simple signal functions

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
    {"r_multiple": 0.8, "close_pct": 0.5}, # Close 50% at 0.8R
]
PARTIAL_TP_MOVE_SL_TO_ENTRY = True # Move SL to entry after first partial TP?
ENABLE_KILL_SWITCH = True       # Enable/disable kill switch mechanism
KILL_SWITCH_MAX_DD_THRESHOLD = 0.25 # [Patch v5.3.5] Max drawdown % before activating kill switch
KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 5 # [Patch] Lower threshold for earlier soft cooldown
MAX_DRAWDOWN_THRESHOLD = 0.15   # [Patch] Reduce drawdown threshold to block orders sooner
logging.info(f"Kill Switch Enabled: {ENABLE_KILL_SWITCH} (DD > {KILL_SWITCH_MAX_DD_THRESHOLD*100:.0f}%, Losses > {KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD})")
logging.info(f"Max Drawdown Threshold (Block New Orders): {MAX_DRAWDOWN_THRESHOLD*100:.0f}%")

# --- Spike Guard & Recovery Mode Configuration ---
logging.debug("Setting Spike Guard & Recovery Mode Configuration...")
ENABLE_SPIKE_GUARD = True       # Enable/disable spike guard filter (mainly London session)
RECOVERY_MODE_CONSECUTIVE_LOSSES = 4 # Consecutive losses to enter recovery mode
RECOVERY_MODE_LOT_MULTIPLIER = 0.5 # Lot size multiplier during recovery mode
logging.info(f"Spike Guard Enabled: {ENABLE_SPIKE_GUARD}")
logging.info(f"Recovery Mode Enabled: Losses >= {RECOVERY_MODE_CONSECUTIVE_LOSSES}, Lot Multiplier: {RECOVERY_MODE_LOT_MULTIPLIER}")

# --- Re-Entry Configuration ---
logging.debug("Setting Re-Entry Configuration...")
USE_REENTRY = True              # Enable/disable re-entry logic
REENTRY_COOLDOWN_BARS = 1       # Cooldown (in bars) after TP before allowing re-entry
REENTRY_MIN_PROBA_THRESH = 0.5 # Minimum ML probability threshold for re-entry (uses META_MIN_PROBA_THRESH)
logging.info(f"Re-Entry Enabled: {USE_REENTRY} (Cooldown: {REENTRY_COOLDOWN_BARS} bars, Threshold: {REENTRY_MIN_PROBA_THRESH})")

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
FORCED_ENTRY_MIN_GAIN_Z_ABS = 1.0 # Min absolute Gain_Z required for FE
FORCED_ENTRY_ALLOWED_REGIMES = ["Normal", "Breakout", "StrongTrend"] # Allowed patterns for FE
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
PATTERN_BREAKOUT_Z_THRESH = 2.0 # Z-Score threshold for 'Breakout' pattern
PATTERN_REVERSAL_BODY_RATIO = 0.5 # Current/Previous body ratio for 'Reversal' pattern
PATTERN_STRONG_TREND_Z_THRESH = 1.0 # Z-Score threshold for 'StrongTrend' pattern
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
