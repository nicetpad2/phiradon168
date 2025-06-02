# === START OF PART 1/12 ===

# -*- coding: utf-8 -*-
# <<< เพิ่ม Encoding declaration สำหรับอักษรไทย >>>

# ==============================================================================
# === PART 1: Setup & Configuration (v4.8.2) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added basic docstrings/comments >>>
# <<< MODIFIED v4.8.1: Updated versioning for comprehensive fixes based on prompt >>>
# <<< MODIFIED v4.8.2: Updated versioning, log filename, and added global df_m15_dt declaration >>>
import logging
import subprocess
import sys
import os
import time
import warnings
import json
import math
import random
from collections import Counter, defaultdict
from joblib import load, dump as joblib_dump
import traceback
import pandas as pd
import numpy as np
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
import requests # For Font Download

# --- Logging Configuration ---
# กำหนดค่าพื้นฐานสำหรับการ Logging
# สามารถปรับ level, format, และ filename ได้ตามต้องการ
LOG_FILENAME = 'gold_ai_v4.8.2.log' # <<< MODIFIED v4.8.2: Updated log filename
logging.basicConfig(
    level=logging.INFO, # ระดับ Log เริ่มต้น (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8'), # บันทึกลงไฟล์ (เขียนทับทุกครั้งที่รัน)
        logging.StreamHandler(sys.stdout) # แสดงผลทาง Console ด้วย
    ]
)
logging.info("--- (Start) Gold AI v4.8.2 ---") # <<< MODIFIED v4.8.2: Updated version
logging.info("--- กำลังโหลดไลบรารีและตรวจสอบ Dependencies ---")

# --- Library Installation & Checks ---
# Helper function to check and log library version
def log_library_version(library_name, library_object):
    """Logs the version of the imported library."""
    try:
        version = getattr(library_object, '__version__', 'N/A')
        logging.info(f"   (Info) Using {library_name} version: {version}")
    except Exception as e:
        logging.warning(f"   (Warning) Could not retrieve {library_name} version: {e}")

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
        tqdm = None # Set tqdm to None if installation fails

# ta library
try:
    import ta
    logging.debug("ta library already installed.")
    log_library_version("TA", ta) # Log version
except ImportError:
    logging.info("   กำลังติดตั้งไลบรารี ta...")
    try:
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", "ta", "-q"],
            check=True, capture_output=True, text=True,
        )
        logging.debug(f"   ผลการติดตั้ง ta: ...{process.stdout[-200:]}")
        import ta
        logging.info("   (Success) ติดตั้ง ta สำเร็จ.")
        log_library_version("TA", ta) # Log version after install
    except Exception as e_install:
        logging.error(f"   (Error) ไม่สามารถติดตั้ง ta: {e_install}", exc_info=True)
        ta = None # Set ta to None if installation fails

# Optuna library
try:
    import optuna
    logging.debug("Optuna library already installed.")
    log_library_version("Optuna", optuna)
    # Consider setting verbosity later if needed
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
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
        logging.error(f"   (Error) ไม่สามารถติดตั้ง optuna: {e_install}. Hyperparameter Optimization จะไม่ทำงาน.", exc_info=True)
        optuna = None # Set optuna to None if installation fails

# XGBoost (Removed in v3.6.6)
XGBClassifier = None
logging.debug("XGBoost is not used in this version.")

# CatBoost library
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
        logging.info(f"   (Success) ติดตั้ง catboost สำเร็จ (เวอร์ชัน: {catboost.__version__}).")
        try:
            from catboost.utils import get_gpu_device_count
            gpu_count_post = get_gpu_device_count()
            logging.info(f"   (Info) ตรวจสอบจำนวน GPU สำหรับ CatBoost (หลังติดตั้ง): {gpu_count_post}")
        except Exception as e_cb_gpu_check_post:
            logging.warning(f"   (Warning) ไม่สามารถตรวจสอบจำนวน GPU ของ CatBoost (หลังติดตั้ง): {e_cb_gpu_check_post}")
    except Exception as e_cat_install:
        logging.error(f"   (Error) ไม่สามารถติดตั้ง catboost: {e_cat_install}. CatBoost models และ SHAP อาจไม่ทำงาน.", exc_info=True)
        CatBoostClassifier = None; Pool = None; catboost = None

# psutil library
try:
    import psutil
    logging.debug("psutil library already installed.")
    log_library_version("psutil", psutil)
except ImportError:
    logging.info("   กำลังติดตั้ง psutil สำหรับตรวจสอบ RAM...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil", "-q"], check=True)
        import psutil
        logging.info("   (Success) ติดตั้ง psutil สำเร็จ.")
        log_library_version("psutil", psutil)
    except Exception as e_install:
        logging.error(f"   (Error) ไม่สามารถติดตั้ง psutil: {e_install}", exc_info=True)
        psutil = None

# SHAP library
try:
    import shap
    logging.debug("shap library already installed.")
    log_library_version("SHAP", shap)
except ImportError:
    logging.info("   กำลังติดตั้งไลบรารี shap...")
    try:
        logging.info("      (การติดตั้ง SHAP อาจใช้เวลาสักครู่...)")
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", "shap", "-q"],
            check=True, capture_output=True, text=True,
        )
        logging.debug(f"   ผลการติดตั้ง shap: ...{process.stdout[-200:]}")
        import shap
        logging.info("   (Success) ติดตั้ง shap สำเร็จ.")
        log_library_version("SHAP", shap)
    except Exception as e_shap_install:
        logging.error(f"   (Error) ไม่สามารถติดตั้ง shap: {e_shap_install}. การวิเคราะห์ SHAP จะถูกข้ามไป.", exc_info=True)
        shap = None

# GPUtil library (Optional for Resource Monitor)
try:
    import GPUtil
    logging.debug("GPUtil library already installed.")
except ImportError:
    logging.info("   กำลังติดตั้ง GPUtil สำหรับตรวจสอบ GPU (Optional)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "gputil", "-q"], check=True)
        import GPUtil
        logging.info("   (Success) ติดตั้ง GPUtil สำเร็จ.")
    except Exception as e_install:
        logging.warning(f"   (Warning) ไม่สามารถติดตั้ง GPUtil: {e_install}. ฟังก์ชัน show_system_status อาจไม่ทำงาน.")
        GPUtil = None

# --- Colab/Drive Setup ---
try:
    import google.colab
    IN_COLAB = True
    logging.info("Running in Google Colab environment.")
except ImportError:
    IN_COLAB = False
    logging.info("Not running in Google Colab environment.")

if IN_COLAB:
    from google.colab import drive
    try:
        logging.info("Attempting to mount Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        logging.info("Google Drive mounted successfully.")
    except Exception as e_drive:
        logging.error(f"Failed to mount Google Drive: {e_drive}", exc_info=True)
else:
    class DummyDrive:
        def mount(self, *args, **kwargs):
            logging.info("   (Info) ข้ามการ Mount Google Drive (ไม่ได้อยู่ใน Colab).")
    drive = DummyDrive()
    drive.mount()

# --- GPU Acceleration Setup (Optional) ---
USE_GPU_ACCELERATION = True
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
                try: pynvml.nvmlShutdown()
                except: pass
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
        try: pynvml.nvmlShutdown()
        except: pass
        nvml_handle = None
    USE_GPU_ACCELERATION = False
logging.info(f"   สถานะการเร่งความเร็วด้วย GPU: {USE_GPU_ACCELERATION}")

# --- GPU/RAM Utilization Helper Function ---
def print_gpu_utilization(context=""):
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
            gpu_util_str = "NVML Err"; gpu_mem_str = f"NVML Err: {e_gpu_mon}"
            logging.warning(f"NVML Error during GPU monitoring: {e_gpu_mon}. Disabling pynvml monitoring.")
            try: pynvml.nvmlShutdown()
            except: pass
            nvml_handle = None
            pynvml = None
        except Exception as e_gpu_mon_other:
            gpu_util_str = "Err"; gpu_mem_str = f"Err: {e_gpu_mon_other}"
            logging.error(f"Unexpected error during GPU monitoring: {e_gpu_mon_other}", exc_info=True)
            try: pynvml.nvmlShutdown()
            except: pass
            nvml_handle = None
            pynvml = None
    elif USE_GPU_ACCELERATION and not pynvml:
        gpu_util_str = "pynvml N/A"; gpu_mem_str = "pynvml N/A"
    elif not USE_GPU_ACCELERATION:
        gpu_util_str = "Disabled"; gpu_mem_str = "Disabled"

    if psutil:
        try:
            ram_info = psutil.virtual_memory()
            ram_str = f"{ram_info.percent:.1f}% ({ram_info.used // 1024**2}MB / {ram_info.total // 1024**2}MB)"
        except Exception as e_ram_mon:
            ram_str = f"Error: {e_ram_mon}"
            logging.error(f"Error getting RAM info: {e_ram_mon}", exc_info=True)
    else:
        ram_str = "psutil N/A"

    logging.info(f"[{context}] GPU Util: {gpu_util_str} | Mem: {gpu_mem_str} | RAM: {ram_str}")

# --- [Optional] System Status Monitor using GPUtil ---
def show_system_status(context=""):
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
# === CONFIGURATION (v4.8.2) ===
# ==============================================================================
logging.info("Loading Global Configuration Settings...")
OUTPUT_BASE_DIR = "/content/drive/MyDrive/new"
OUTPUT_DIR_NAME = "outputgpt_v4.8.2"
DATA_FILE_PATH_M15 = "/content/drive/MyDrive/new/XAUUSD_M15.csv"
DATA_FILE_PATH_M1 = "/content/drive/MyDrive/new/XAUUSD_M1.csv"
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
MAX_CONCURRENT_ORDERS = 5       # Max concurrent orders per side (BUY/SELL)
MAX_HOLDING_BARS = 24           # Max bars an order can be held
COMMISSION_PER_001_LOT = 0.10   # Commission per 0.01 lot (USD)
SPREAD_POINTS = 2.0             # Fixed spread in points
MIN_SLIPPAGE_POINTS = -5.0      # Minimum slippage in points (negative means better price)
MAX_SLIPPAGE_POINTS = -1.0      # Maximum slippage in points (negative means better price)

# --- Entry/Exit Logic Parameters ---
logging.debug("Setting Entry/Exit Logic Parameters...")
MIN_SIGNAL_SCORE_ENTRY = 2.0    # Minimum signal score required to open an order
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
SESSION_TIMES_UTC = {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)} # Session times in UTC
logging.debug(f"Session Times (UTC): {SESSION_TIMES_UTC}")

# --- Fold-Specific Configuration ---
# Allows overriding parameters for specific walk-forward folds
logging.debug("Setting Fold-Specific Configuration...")
ENTRY_CONFIG_PER_FOLD = {
    # Fold Index: {Config Dictionary}
    0: {"sl_multiplier": 2.0, "gain_z_thresh": 0.3, "cooldown_sec": 0, "min_signal_score": MIN_SIGNAL_SCORE_ENTRY, },
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
KILL_SWITCH_MAX_DD_THRESHOLD = 0.20 # Max drawdown % before activating kill switch
KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 7 # Max consecutive losses before activating kill switch
MAX_DRAWDOWN_THRESHOLD = 0.30   # Max drawdown % threshold to block new orders (e.g., 30%)
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
REENTRY_MIN_PROBA_THRESH = 0.55 # Minimum ML probability threshold for re-entry (uses META_MIN_PROBA_THRESH)
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
DRIFT_WASSERSTEIN_THRESHOLD = 0.1 # Wasserstein Distance threshold for drift alert
DRIFT_TTEST_ALPHA = 0.05        # Alpha level for T-test drift detection
SIGNIFICANCE_LEVEL = 0.05       # (Not directly used, kept for potential analysis)
M1_FEATURES_FOR_DRIFT = []      # Will be populated in clean_m1_data (Part 5)
MAX_NAT_RATIO_THRESHOLD = 0.05  # Max allowed NaT ratio after datetime parsing

# --- Dynamic Adjustment Configuration ---
logging.debug("Setting Dynamic Adjustment Configuration...")
DYNAMIC_GAINZ_DRIFT_THRESHOLD = 0.10 # Wasserstein threshold on Gain_Z to trigger adjustment
DYNAMIC_GAINZ_ADJUSTMENT = 0.1  # Amount to add to Gain_Z entry threshold on high drift
DYNAMIC_RISK_DD_THRESHOLD = 12.0 # Drawdown % to trigger risk reduction (not used)
DYNAMIC_RISK_REDUCTION_FACTOR = 0.7 # Factor to reduce risk by on high DD (not used)

logging.info("Part 2: Core Parameters & Strategy Settings Loaded.")
# === END OF PART 2/12 ===
# -*- coding: utf-8 -*-
# <<< เพิ่ม Encoding declaration สำหรับอักษรไทย (ควรอยู่บรรทัดแรกหรือสองของไฟล์) >>>

# ==============================================================================
# -*- coding: utf-8 -*-
# <<< เพิ่ม Encoding declaration สำหรับอักษรไทย (ควรอยู่บรรทัดแรกหรือสองของไฟล์) >>>

# ==============================================================================
# === START OF PART 3/12 ===
# ==============================================================================
# === PART 3: Helper Functions (Setup, Utils, Font, Config) (v4.8.8 - Patch 26.10 Applied) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, improved font setup robustness >>>
# <<< MODIFIED v4.7.9 (Post-Error): Corrected simple_converter for np.inf, then updated to 'Infinity' string with improved docstring >>>
# <<< MODIFIED v4.8.1: Added ValueError handling in parse_datetime_safely, refined simple_converter for NINF >>>
# <<< MODIFIED v4.8.2: Enhanced ValueError logging in parse_datetime_safely, refined simple_converter for NaN/Inf/NA/NaT and other types >>>
# <<< MODIFIED v4.8.3: Ensured simple_converter correctly handles np.inf/-np.inf to string "Infinity"/"-Infinity" and other types for JSON.
#                      Corrected datetime import and usage. Re-indented and reviewed. Added Part Markers. >>>
# <<< MODIFIED v4.8.4: Added load_app_config function and updated versioning. >>>
# <<< MODIFIED v4.8.5: Added safe_get_global function definition. >>>
# <<< MODIFIED v4.8.8 (Patch 10): Refined safe_set_datetime to proactively handle column dtype to prevent FutureWarning. >>>
# <<< MODIFIED v4.8.8 (Patch 26.1): Corrected safe_set_datetime assignment to use pd.to_datetime(val).astype("datetime64[ns]") for robust dtype handling. >>>
# <<< MODIFIED v4.8.8 (Patch 26.3.1): Applied [PATCH A] to safe_set_datetime as per user prompt. >>>
# <<< MODIFIED v4.8.8 (Patch 26.4.1): Unified [PATCH A] for safe_set_datetime. >>>
# <<< MODIFIED v4.8.8 (Patch 26.5.1): Applied final [PATCH A] for safe_set_datetime from user prompt. >>>
# <<< MODIFIED v4.8.8 (Patch 26.7): Applied fix for FutureWarning in safe_set_datetime by ensuring column dtype is datetime64[ns] before assignment. >>>
# <<< MODIFIED v4.8.8 (Patch 26.8): Applied model_diagnostics_unit recommendation to safe_set_datetime for robust dtype handling. >>>
# <<< MODIFIED v4.8.8 (Patch 26.10): Further refined safe_set_datetime to more aggressively ensure column dtype is datetime64[ns] before assignment. >>>
import logging
import os
import sys
import subprocess
import traceback
import pandas as pd
import numpy as np
import json
import gzip
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from IPython import get_ipython
import requests
import datetime # <<< ENSURED Standard import 'import datetime'

# --- Helper for Safe Global Access (Defined *before* use in other parts) ---
# <<< [Patch] ADDED v4.8.5: Function definition moved here >>>
def safe_get_global(var_name, default_value):
    """
    Safely retrieves a global variable from the current module's scope.

    Args:
        var_name (str): The name of the global variable to retrieve.
        default_value: The value to return if the global variable is not found.

    Returns:
        The value of the global variable or the default value.
    """
    try:
        # Use globals() which returns a dictionary representing the current global symbol table
        return globals().get(var_name, default_value)
    except Exception as e:
        # Log an error if there's an unexpected issue accessing globals()
        logging.error(f"   (Error) Unexpected error in safe_get_global for '{var_name}': {e}", exc_info=True)
        return default_value
# <<< End of [Patch] ADDED v4.8.5 >>>

# --- Directory Setup Helper ---
def setup_output_directory(base_dir, dir_name):
    """
    Creates the output directory if it doesn't exist and checks write permissions.

    Args:
        base_dir (str): The base directory path.
        dir_name (str): The name of the output directory to create within the base directory.

    Returns:
        str: The full path to the created/verified output directory.

    Raises:
        SystemExit: If the directory cannot be created or written to.
    """
    output_path = os.path.join(base_dir, dir_name)
    logging.info(f"   (Setup) กำลังตรวจสอบ/สร้าง Output Directory: {output_path}")
    try:
        os.makedirs(output_path, exist_ok=True)
        logging.info(f"      -> Directory exists or was created.")
        # Test write permissions
        test_file_path = os.path.join(output_path, ".write_test")
        with open(test_file_path, "w", encoding='utf-8') as f:
            f.write("test")
        os.remove(test_file_path)
        logging.info(f"      -> การเขียนไฟล์ทดสอบสำเร็จ.")
        return output_path
    except OSError as e:
        logging.error(f"   (Error) ไม่สามารถสร้างหรือเขียนใน Output Directory '{output_path}': {e}", exc_info=True)
        sys.exit(f"   ออก: ปัญหาการเข้าถึง Output Directory ({output_path}).")
    except Exception as e:
        logging.error(f"   (Error) เกิดข้อผิดพลาดที่ไม่คาดคิดระหว่างตั้งค่า Output Directory '{output_path}': {e}", exc_info=True)
        sys.exit(f"   ออก: ข้อผิดพลาดร้ายแรงในการตั้งค่า Output Directory ({output_path}).")

# --- Font Setup Helpers ---
def set_thai_font(font_name="Loma"):
    """
    Attempts to set the specified Thai font for Matplotlib using findfont.
    Prioritizes specified font, then searches common fallbacks.

    Args:
        font_name (str): The preferred Thai font name. Defaults to "Loma".

    Returns:
        bool: True if a preferred or fallback font was successfully set and tested, False otherwise.
    """
    target_font_path = None
    actual_font_name = None
    # Added more common Thai fonts and ensured uniqueness
    preferred_fonts = [font_name] + ["TH Sarabun New", "THSarabunNew", "Garuda", "Norasi", "Kinnari", "Waree", "Laksaman", "Loma"]
    preferred_fonts = list(dict.fromkeys(preferred_fonts)) # Remove duplicates while preserving order
    logging.info(f"   [Font Check] Searching for preferred fonts: {preferred_fonts}")

    for pref_font in preferred_fonts:
        try:
            found_path = fm.findfont(pref_font, fallback_to_default=False)
            if found_path and os.path.exists(found_path):
                target_font_path = found_path
                prop = fm.FontProperties(fname=target_font_path)
                actual_font_name = prop.get_name()
                logging.info(f"      -> Found font: '{actual_font_name}' (requested: '{pref_font}') at path: {target_font_path}")
                break
        except ValueError:
            logging.debug(f"      -> Font '{pref_font}' not found by findfont.")
        except Exception as e_find: # Catch more general exceptions during findfont
            logging.warning(f"      -> Error finding font '{pref_font}': {e_find}")

    if target_font_path and actual_font_name:
        try:
            plt.rcParams['font.family'] = actual_font_name
            plt.rcParams['axes.unicode_minus'] = False # Important for correct display of minus sign
            logging.info(f"   Attempting to set default font to '{actual_font_name}'.")

            # Test plot to confirm font rendering
            fig_test, ax_test = plt.subplots(figsize=(0.5, 0.5)) # Small test figure
            ax_test.set_title(f"ทดสอบ ({actual_font_name})", fontname=actual_font_name)
            plt.close(fig_test) # Close the test figure immediately
            logging.info(f"      -> Font '{actual_font_name}' set and tested successfully.")
            return True
        except Exception as e_set:
            logging.warning(f"      -> (Warning) Font '{actual_font_name}' set, but test failed: {e_set}")
            # Attempt to revert to a known safe default if setting the Thai font fails
            try:
                plt.rcParams['font.family'] = 'DejaVu Sans' # A common fallback
                logging.info("         -> Reverted to 'DejaVu Sans' due to test failure.")
            except Exception as e_revert:
                logging.warning(f"         -> Failed to revert font to DejaVu Sans: {e_revert}")
            return False
    else:
        logging.warning(f"   (Warning) Could not find any suitable Thai fonts ({preferred_fonts}) using findfont.")
        return False

def setup_fonts(output_dir=None): # output_dir is not used in current implementation but kept for potential future use
    """
    Sets up Thai fonts for Matplotlib plots.
    Attempts to find preferred fonts, installs 'fonts-thai-tlwg' on Colab if needed.
    """
    logging.info("\n(Processing) Setting up Thai font for plots...")
    font_set_successfully = False
    preferred_font_name = "TH Sarabun New" # Prioritize this font

    try:
        ipython = get_ipython()
        in_colab = ipython is not None and 'google.colab' in str(ipython)

        logging.info("   Attempting to set font directly using findfont...")
        font_set_successfully = set_thai_font(preferred_font_name)

        if not font_set_successfully and in_colab:
            logging.info("\n   Preferred font not found. Attempting installation via apt-get (Colab)...")
            try:
                logging.info("      Installing Thai fonts (fonts-thai-tlwg)... This might take a moment.")
                # Update package list quietly
                apt_update_process = subprocess.run(
                    ["apt-get", "update", "-qq"],
                    check=False, capture_output=True, text=True, timeout=120 # Added timeout
                )
                if apt_update_process.returncode != 0:
                    logging.warning(f"      (Warning) apt-get update failed (Code: {apt_update_process.returncode}): {apt_update_process.stderr[:200]}...")

                # Install Thai fonts quietly
                apt_install_process = subprocess.run(
                    ["apt-get", "install", "-y", "-qq", "fonts-thai-tlwg"],
                    check=False, capture_output=True, text=True, timeout=180 # Added timeout
                )

                if apt_install_process.returncode == 0:
                    logging.info("      (Success) apt-get install fonts-thai-tlwg potentially completed.")
                    logging.info("      Rebuilding Matplotlib font cache...")
                    try:
                        fm._load_fontmanager(try_read_cache=False) # Force rebuild
                        logging.info("      Font cache rebuilt. Attempting to set font again...")
                        font_set_successfully = set_thai_font(preferred_font_name)
                        if not font_set_successfully: # Try another common one if preferred still not found
                            font_set_successfully = set_thai_font("Loma")

                        if font_set_successfully:
                            logging.info("      (Success) Thai font set after installation and cache rebuild.")
                        else:
                            logging.warning("      (Warning) Thai font still not set after installation. A manual Colab Runtime Restart might be needed.")
                            logging.warning("      *****************************************************")
                            logging.warning("      *** Please RESTART RUNTIME now for Matplotlib     ***")
                            logging.warning("      *** to recognize the new fonts if plots fail.     ***")
                            logging.warning("      *** (เมนู Runtime -> Restart runtime...)         ***")
                            logging.warning("      *****************************************************")
                    except Exception as e_cache:
                        logging.error(f"      (Error) Failed to rebuild font cache or set font after install: {e_cache}", exc_info=True)
                else:
                    logging.warning(f"      (Warning) apt-get install failed (Code: {apt_install_process.returncode}): {apt_install_process.stderr[:200]}...")
            except subprocess.TimeoutExpired:
                logging.error("      (Error) Timeout during apt-get font installation.")
            except Exception as e_generic_install: # Catch any other installation errors
                logging.error(f"      (Error) General error during font installation attempt: {e_generic_install}", exc_info=True)

        # If still not set, try other fallbacks
        if not font_set_successfully:
            fallback_fonts = ["Loma", "Garuda", "Norasi", "Kinnari", "Waree", "THSarabunNew"] # Ensure THSarabunNew is tried again if initial fails
            logging.info(f"\n   Trying fallbacks ({', '.join(fallback_fonts)})...")
            for fb_font in fallback_fonts:
                if set_thai_font(fb_font):
                    font_set_successfully = True
                    break

        if not font_set_successfully:
            logging.critical("\n   (CRITICAL WARNING) Could not set any preferred Thai font. Plots WILL NOT render Thai characters correctly.")
        else:
            logging.info("\n   (Info) Font setup process complete.")

    except Exception as e:
        logging.error(f"   (Error) Critical error during font setup: {e}", exc_info=True)

# --- Data Loading Helper ---
def safe_load_csv_auto(file_path):
    """
    Loads CSV or .csv.gz file using pandas, automatically handling gzip compression.

    Args:
        file_path (str): The path to the CSV or gzipped CSV file.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, an empty DataFrame if the file
                              is empty, or None if loading fails.
    """
    read_csv_kwargs = {"index_col": 0, "parse_dates": False, "low_memory": False}
    logging.info(f"      (safe_load) Attempting to load: {os.path.basename(file_path)}")

    if not isinstance(file_path, str) or not file_path:
        logging.error("         (Error) Invalid file path provided to safe_load_csv_auto.")
        return None
    if not os.path.exists(file_path):
        logging.error(f"         (Error) File not found: {file_path}")
        return None

    try:
        if file_path.lower().endswith(".gz"):
            logging.debug("         -> Detected .gz extension, using gzip.")
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return pd.read_csv(f, **read_csv_kwargs)
        else:
            logging.debug("         -> No .gz extension, using standard pd.read_csv.")
            return pd.read_csv(file_path, **read_csv_kwargs)
    except pd.errors.EmptyDataError:
        logging.warning(f"         (Warning) File is empty: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"         (Error) Failed to load file '{os.path.basename(file_path)}': {e}", exc_info=True)
        return None

# --- JSON Serialization Helper ---
def simple_converter(o):
    """
    Converts numpy/pandas types for JSON serialization, handling NaN/Inf/other non-serializable types.
    Returns "Infinity" or "-Infinity" for np.inf/np.NINF to comply with JSON standard.
    Handles np.nan and pd.NA/pd.NaT by returning None.
    """
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        if np.isnan(o):
            return None
        if np.isinf(o):
            return "Infinity" if o > 0 else "-Infinity"
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, np.bool_):
        return bool(o)
    if pd.isna(o):
        return None
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    try:
        if isinstance(o, (str, bool, list, dict, type(None))):
            json.dumps(o) # Test if directly serializable
            return o
        # Fallback for other types not directly serializable by default json.dumps
        str_representation = str(o)
        logging.debug(f"simple_converter: Type {type(o)} not directly serializable. Converting to string: '{str_representation[:100]}...'")
        return str_representation
    except TypeError:
        # This TypeError might occur during json.dumps test if the object is complex
        str_representation_on_error = str(o)
        logging.warning(f"simple_converter: TypeError for type {type(o)}. Using str(): '{str_representation_on_error[:100]}...'")
        return str_representation_on_error
    except Exception as e:
        # Catch any other unexpected errors during conversion or stringification
        str_representation_on_general_error = str(o)
        logging.error(f"simple_converter: Unexpected error for type {type(o)}: {e}. Using str(): '{str_representation_on_general_error[:100]}...'", exc_info=True)
        return str_representation_on_general_error


# --- Configuration Loading Helper ---
def load_app_config(config_path="config_main.json"):
    """Loads application configuration from a JSON file."""
    try:
        # Attempt to find the config file relative to the script's location first
        # This is more robust when the script is run from different working directories.
        script_dir = ""
        try:
            # __file__ is defined when running a script directly
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback if __file__ is not defined (e.g., in some interactive environments like Jupyter without a file context)
            # Tries current working directory
            script_dir = os.getcwd()
            logging.debug(f"   (Config) __file__ not defined, using CWD: {script_dir} for config lookup.")


        potential_path = os.path.join(script_dir, config_path)
        actual_config_path = None

        if os.path.exists(potential_path):
            actual_config_path = potential_path
        elif os.path.exists(config_path): # Fallback to current working directory if not found with script
            actual_config_path = config_path
            logging.debug(f"   (Config) Config not found at '{potential_path}', trying '{config_path}' in CWD.")

        if actual_config_path is None: # If still not found
            raise FileNotFoundError(f"Configuration file '{config_path}' not found in script directory ('{script_dir}') or CWD.")


        with open(actual_config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        logging.info(f"   (Config) Successfully loaded configuration from: {actual_config_path}")
        return config_data
    except FileNotFoundError:
        logging.error(f"   (Config Error) Configuration file '{config_path}' not found. Using default script values.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"   (Config Error) Error decoding JSON from configuration file: {config_path}. Using default script values.")
        return {}
    except Exception as e:
        logging.error(f"   (Config Error) Failed to load configuration from '{config_path}': {e}", exc_info=True)
        return {}

# --- Datetime Setting Helper (Corrected for FutureWarning) ---
# <<< [Patch] MODIFIED v4.8.8 (Patch 26.10): Applied model_diagnostics_unit recommendation with refined dtype handling. >>>
def safe_set_datetime(df, idx, col, val):
    """
    Safely assigns datetime value to DataFrame, ensuring column dtype is datetime64[ns].
    [PATCH 26.10] Applied: Ensures column dtype is datetime64[ns] before assignment
    by initializing or converting the entire column if necessary.
    """
    try:
        # Convert the input value to a pandas Timestamp or NaT
        dt_value = pd.to_datetime(val, errors='coerce')

        # Ensure the column exists and has the correct dtype BEFORE assignment
        if col not in df.columns:
            logging.debug(f"   [Patch 26.10] safe_set_datetime: Column '{col}' not found. Creating with dtype 'datetime64[ns]'.")
            # Initialize the entire column with NaT and correct dtype
            # This helps prevent the FutureWarning when assigning the first Timestamp/NaT
            df[col] = pd.Series(dtype='datetime64[ns]', index=df.index)
        elif df[col].dtype != 'datetime64[ns]':
            logging.debug(f"   [Patch 26.10] safe_set_datetime: Column '{col}' has dtype '{df[col].dtype}'. Forcing conversion to 'datetime64[ns]'.")
            try:
                # Attempt to convert the existing column to datetime64[ns]
                # This is important if the column was, e.g., object or float due to prior NaNs
                # Using pd.to_datetime on the series first handles mixed types better before astype
                current_col_values = pd.to_datetime(df[col], errors='coerce')
                df[col] = current_col_values.astype('datetime64[ns]')
            except Exception as e_conv_col:
                logging.warning(f"   [Patch 26.10] safe_set_datetime: Force conversion of column '{col}' to datetime64[ns] failed ({e_conv_col}). Re-creating column with NaT.")
                # If conversion fails (e.g., mixed types that can't be coerced easily),
                # re-create the column with the correct dtype. This might lose existing data in the column if it was incompatible.
                df[col] = pd.Series(dtype='datetime64[ns]', index=df.index)

        # Now assign the value (which is already a Timestamp or NaT)
        if idx in df.index:
            # dt_value is already pd.Timestamp or pd.NaT
            # df[col] should now have dtype datetime64[ns]
            df.loc[idx, col] = dt_value
            logging.debug(f"   [Patch 26.10] safe_set_datetime: Assigned '{dt_value}' (type: {type(dt_value)}) to '{col}' at index {idx}. Column dtype after assign: {df[col].dtype}")
        else:
            logging.warning(f"   safe_set_datetime: Index '{idx}' not found in DataFrame. Cannot set value for column '{col}'.")

    except Exception as e:
        logging.error(f"   (Error) safe_set_datetime: Failed to assign '{val}' (type: {type(val)}) to '{col}' at {idx}: {e}", exc_info=True)
        # Fallback to NaT if any error occurs during assignment
        try:
            if idx in df.index:
                if col not in df.columns or df[col].dtype != 'datetime64[ns]':
                    # Ensure column exists with a datetime-compatible dtype if creating/fixing it during fallback
                    df[col] = pd.Series(dtype='datetime64[ns]', index=df.index)
                df.loc[idx, col] = pd.NaT
            else:
                logging.warning(f"   safe_set_datetime: Index '{idx}' not found during fallback NaT assignment for column '{col}'.")
        except Exception as e_fallback:
            logging.error(f"   (Error) safe_set_datetime: Failed to assign NaT as fallback for '{col}' at index {idx}: {e_fallback}")
# <<< End of [Patch] MODIFIED v4.8.8 (Patch 26.10) >>>

logging.info("Part 3: Helper Functions Loaded (v4.8.8 Patch 26.10 Applied).")
# ==============================================================================
# === END OF PART 3/12 ===
# ==============================================================================
# === START OF PART 4/12 ===

# ==============================================================================
# === PART 4: Data Loading & Initial Preparation (v4.8.3) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, refined error handling, added dtype specification >>>
# <<< MODIFIED v4.8.1: Enhanced prepare_datetime for empty df, SystemExit, NaT handling; added ValueError to parse_datetime_safely (via Part 3) >>>
# <<< MODIFIED v4.8.2: Enhanced prepare_datetime for empty df after NaT drop, and ensured datetime_original conversion before set_index >>>
# <<< MODIFIED v4.8.3: Removed redundant 'from datetime import datetime', ensured usage of 'datetime.datetime.now()'. Re-indented. >>>
import logging
import os
import sys
import pandas as pd
import numpy as np
import warnings
import traceback
# from datetime import datetime # <<< REMOVED: Should use global 'datetime' module imported earlier
import gc
# Ensure 'datetime' module is available from global imports (e.g., Part 3 or top of file)
# import datetime # This would be redundant if already imported globally

# Ensure global configurations are accessible if run independently
try:
    MAX_NAT_RATIO_THRESHOLD
except NameError:
    logging.warning("MAX_NAT_RATIO_THRESHOLD not defined globally, using default 0.05")
    MAX_NAT_RATIO_THRESHOLD = 0.05

# --- Data Loading Function ---
def load_data(file_path, timeframe_str="", price_jump_threshold=0.10, nan_threshold=0.05, dtypes=None):
    """
    Loads data from a CSV file, performs basic validation and data quality checks.

    Args:
        file_path (str): Path to the CSV file.
        timeframe_str (str): Identifier for the timeframe (e.g., "M15", "M1") for logging.
        price_jump_threshold (float): Threshold for price percentage change between bars
                                      to be considered anomalous. Defaults to 0.10 (10%).
        nan_threshold (float): Maximum acceptable proportion of NaN values in price columns.
                               Defaults to 0.05 (5%).
        dtypes (dict, optional): Dictionary specifying data types for columns during loading.
                                 Defaults to None (pandas infers types).

    Returns:
        pd.DataFrame: The loaded and initially validated DataFrame.

    Raises:
        SystemExit: If critical errors occur (e.g., file not found, essential columns missing).
    """
    logging.info(f"(Loading) กำลังโหลดข้อมูล {timeframe_str} จาก: {file_path}")
    if not os.path.exists(file_path):
        logging.critical(f"(Error) ไม่พบไฟล์: {file_path}")
        sys.exit(f"ออก: ไม่พบไฟล์ข้อมูล {timeframe_str} ที่ {file_path}")

    try:
        try:
            df_pd = pd.read_csv(file_path, low_memory=False, dtype=dtypes)
            logging.info(f"   ไฟล์ดิบ {timeframe_str}: {df_pd.shape[0]} แถว")
        except pd.errors.ParserError as e_parse:
            logging.critical(f"(Error) ไม่สามารถ Parse ไฟล์ CSV '{file_path}': {e_parse}")
            sys.exit(f"ออก: ปัญหาการ Parse ไฟล์ CSV {timeframe_str}")
        except Exception as e_read:
            logging.critical(f"(Error) ไม่สามารถอ่านไฟล์ CSV '{file_path}': {e_read}", exc_info=True)
            sys.exit(f"ออก: ปัญหาการอ่านไฟล์ CSV {timeframe_str}")

        required_cols_base = ["Date", "Timestamp", "Open", "High", "Low", "Close"]
        required_cols_check = list(dtypes.keys()) if dtypes else required_cols_base
        required_cols_check = sorted(list(set(required_cols_check + required_cols_base)))
        missing_req = [col for col in required_cols_check if col not in df_pd.columns]
        if missing_req:
            logging.critical(f"(Error) ขาดคอลัมน์: {missing_req} ใน {file_path}")
            sys.exit(f"ออก: ขาดคอลัมน์ที่จำเป็นในข้อมูล {timeframe_str}")

        price_cols = ["Open", "High", "Low", "Close"]
        logging.debug(f"   Converting price columns {price_cols} to numeric (if not already specified in dtypes)...")
        for col in price_cols:
            if dtypes is None or col not in dtypes or not pd.api.types.is_numeric_dtype(df_pd[col].dtype):
                df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')

        logging.debug("   [Data Quality] Checking for invalid prices (<= 0)...")
        for col in price_cols:
            invalid_prices = df_pd[pd.notna(df_pd[col]) & (df_pd[col] <= 0)]
            if not invalid_prices.empty:
                logging.warning(f"   (Warning) พบราคาที่ผิดปกติ (<= 0) ในคอลัมน์ '{col}' จำนวน {len(invalid_prices)} แถว. แถวตัวอย่าง:\n{invalid_prices.head()}")

        logging.debug("   [Data Quality] Checking High >= Low consistency...")
        invalid_hl = df_pd[pd.notna(df_pd['High']) & pd.notna(df_pd['Low']) & (df_pd['High'] < df_pd['Low'])]
        if not invalid_hl.empty:
            logging.warning(f"   (Warning) พบราคา High < Low จำนวน {len(invalid_hl)} แถว. แถวตัวอย่าง:\n{invalid_hl.head()}")

        logging.info("   [Data Quality] ตรวจสอบ % NaN ในคอลัมน์ราคา...")
        nan_report = df_pd[price_cols].isnull().mean()
        logging.info(f"      NaN Percentage:\n{nan_report.round(4)}")
        high_nan_cols = nan_report[nan_report > nan_threshold].index.tolist()
        if high_nan_cols:
            logging.warning(f"   (Warning) คอลัมน์ {high_nan_cols} มี NaN เกินเกณฑ์ ({nan_threshold:.1%}).")

        initial_rows = df_pd.shape[0]
        df_pd.dropna(subset=price_cols, inplace=True)
        rows_dropped_nan = initial_rows - df_pd.shape[0]
        if rows_dropped_nan > 0:
            logging.info(f"   ลบ {rows_dropped_nan} แถวที่มีราคาเป็น NaN.")

        logging.info("   [Data Quality] ตรวจสอบ Duplicates (Date & Timestamp)...")
        duplicate_cols = ["Date", "Timestamp"]
        if all(col in df_pd.columns for col in duplicate_cols):
            num_duplicates = df_pd.duplicated(subset=duplicate_cols, keep=False).sum()
            if num_duplicates > 0:
                logging.warning(f"   (Warning) พบ {num_duplicates} แถวที่มี Date & Timestamp ซ้ำกัน. กำลังลบรายการซ้ำ (เก็บรายการแรก)...")
                df_pd.drop_duplicates(subset=duplicate_cols, keep='first', inplace=True)
                logging.info(f"      ขนาดข้อมูลหลังลบ Duplicates: {df_pd.shape[0]} แถว.")
            else:
                logging.debug("      ไม่พบ Duplicates (Date & Timestamp).")
        else:
            logging.warning(f"   (Warning) ขาดคอลัมน์ {duplicate_cols} สำหรับตรวจสอบ Duplicates.")

        logging.info(f"   [Data Quality] ตรวจสอบ Price Jumps (Threshold > {price_jump_threshold:.1%})...")
        if 'Close' in df_pd.columns and len(df_pd) > 1:
            df_pd['Close'] = pd.to_numeric(df_pd['Close'], errors='coerce')
            close_numeric = df_pd['Close'].dropna()
            if len(close_numeric) > 1:
                price_pct_change = close_numeric.pct_change().abs()
                large_jumps = price_pct_change[price_pct_change > price_jump_threshold]
                if not large_jumps.empty:
                    logging.warning(f"   (Warning) พบ {len(large_jumps)} แท่งที่มีการเปลี่ยนแปลงราคา Close เกิน {price_jump_threshold:.1%}:")
                    example_jumps = large_jumps.head()
                    logging.warning(f"      ตัวอย่าง Index และ % Change:\n{example_jumps.round(4).to_string()}")
                else:
                    logging.debug("      ไม่พบ Price Jumps ที่ผิดปกติ.")
                del close_numeric, price_pct_change, large_jumps
                gc.collect()
            else:
                logging.debug("      ข้ามการตรวจสอบ Price Jumps (ข้อมูล Close ไม่พอหลัง dropna).")
        else:
            logging.debug("      ข้ามการตรวจสอบ Price Jumps (ไม่มีข้อมูล Close หรือมีน้อยกว่า 2 แถว).")

        if df_pd.empty:
            logging.warning(f"   (Warning) DataFrame ว่างเปล่าหลังจากลบราคา NaN และ Duplicates ({timeframe_str}).")

        logging.info(f"(Success) โหลดและตรวจสอบข้อมูล {timeframe_str} สำเร็จ: {df_pd.shape[0]} แถว")
        return df_pd

    except SystemExit as se:
        raise se
    except Exception as e:
        logging.critical(f"(Error) ไม่สามารถโหลดข้อมูล {timeframe_str}: {e}\n{traceback.format_exc()}", exc_info=True)
        sys.exit(f"ออก: ข้อผิดพลาดร้ายแรงในการโหลดข้อมูล {timeframe_str}")

# --- Datetime Helper Functions ---
def preview_datetime_format(df, n=5):
    """Displays a preview of the Date + Timestamp string format before conversion."""
    if df is None or df.empty or "Date" not in df.columns or "Timestamp" not in df.columns:
        logging.warning("   [Preview] Cannot preview: DataFrame is empty or missing Date/Timestamp columns.")
        return
    logging.info(f"   [Preview] First {n} Date + Timestamp format examples:")
    try:
        preview_df = df.head(n).copy()
        preview_df["Date"] = preview_df["Date"].astype(str).str.strip()
        preview_df["Timestamp"] = (
            preview_df["Timestamp"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        preview = preview_df.apply(lambda row: f"{row['Date']} {row['Timestamp']}", axis=1)
        logging.info("\n" + preview.to_string(index=False))
        del preview_df, preview
        gc.collect()
    except Exception as e:
        logging.error(f"   [Preview] Error during preview generation: {e}", exc_info=True)

def parse_datetime_safely(datetime_str_series):
    """
    Attempts to parse a Series of datetime strings into datetime objects using multiple formats.

    Args:
        datetime_str_series (pd.Series): Series containing datetime strings.

    Returns:
        pd.Series: Series with parsed datetime objects (dtype datetime64[ns]),
                   or NaT where parsing failed.
    """
    if not isinstance(datetime_str_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    if datetime_str_series.empty:
        logging.debug("      [Parser] Input series is empty, returning empty series.")
        return datetime_str_series

    logging.info("      [Parser] Attempting to parse date/time strings...")
    common_formats = [
        "%Y%m%d %H:%M:%S", "%Y%m%d%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S", "%Y.%m.%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ]
    parsed_results = pd.Series(pd.NaT, index=datetime_str_series.index, dtype="datetime64[ns]")
    remaining_indices = datetime_str_series.index.copy()
    series_to_parse = datetime_str_series.copy()

    for fmt in common_formats:
        if remaining_indices.empty:
            logging.debug("      [Parser] All strings parsed.")
            break
        logging.debug(f"      [Parser] Trying format: '{fmt}' for {len(remaining_indices)} values...")
        try:
            try_parse = pd.to_datetime(
                series_to_parse.loc[remaining_indices], format=fmt, errors='coerce'
            )
            successful_mask_this_attempt = try_parse.notna()
            successful_indices_this_attempt = remaining_indices[successful_mask_this_attempt]
            if not successful_indices_this_attempt.empty:
                parsed_results.loc[successful_indices_this_attempt] = try_parse[successful_mask_this_attempt]
                remaining_indices = remaining_indices.difference(successful_indices_this_attempt)
                logging.info(
                    f"      [Parser] (Success) Format '{fmt}' matched: {len(successful_indices_this_attempt)}. Remaining: {len(remaining_indices)}"
                )
            del try_parse, successful_mask_this_attempt, successful_indices_this_attempt
            gc.collect()
        except ValueError as ve:
            if not remaining_indices.empty:
                first_failed_idx = remaining_indices[0]
                first_failed_str = series_to_parse.get(first_failed_idx, 'N/A')
                logging.warning(f"      [Parser] Invalid format '{fmt}' for string like '{first_failed_str}' (ValueError: {ve}). Trying next format.")
            else:
                logging.warning(f"      [Parser] ValueError with format '{fmt}' but no remaining indices to sample from (ValueError: {ve}).")
            pass
        except Exception as e_fmt:
            logging.warning(f"         -> General error while trying format '{fmt}': {e_fmt}", exc_info=True)
            pass

    if not remaining_indices.empty:
        logging.info(f"      [Parser] Trying general parser for {len(remaining_indices)} remaining values...")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
                try_general = pd.to_datetime(series_to_parse.loc[remaining_indices], errors='coerce')
            successful_mask_general = try_general.notna()
            successful_indices_general = remaining_indices[successful_mask_general]
            if not successful_indices_general.empty:
                parsed_results.loc[successful_indices_general] = try_general[successful_mask_general]
                remaining_indices = remaining_indices.difference(successful_indices_general)
                logging.info(f"         -> (Success) General parser matched: {len(successful_indices_general)}. Remaining: {len(remaining_indices)}")
            del try_general, successful_mask_general, successful_indices_general
            gc.collect()
        except Exception as e_gen:
            logging.warning(f"         -> General parser error: {e_gen}", exc_info=True)

    final_nat_count = parsed_results.isna().sum()
    if final_nat_count > 0:
        logging.warning(f"      [Parser] Could not parse {final_nat_count} date/time strings.")
        failed_strings_log = series_to_parse[parsed_results.isna()].head(5)
        logging.warning(f"         Example failed strings:\n{failed_strings_log.to_string()}")
    logging.info("      [Parser] (Finished) Date/time parsing complete.")
    del series_to_parse, remaining_indices
    gc.collect()
    return parsed_results

def prepare_datetime(df_pd, timeframe_str=""):
    """
    Prepares the DatetimeIndex for the DataFrame, handling Buddhist Era conversion
    and NaT values. Sets the prepared datetime as the DataFrame index.

    Args:
        df_pd (pd.DataFrame): Input DataFrame with 'Date' and 'Timestamp' columns.
        timeframe_str (str): Identifier for the timeframe (e.g., "M15", "M1") for logging.

    Returns:
        pd.DataFrame: DataFrame with a sorted DatetimeIndex, or raises SystemExit on critical errors.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        SystemExit: If essential columns are missing, all datetimes fail to parse,
                    or other critical errors occur.
    """
    logging.info(f"(Processing) กำลังเตรียม Datetime Index ({timeframe_str})...")
    if not isinstance(df_pd, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df_pd.empty:
        logging.warning(f"   (Warning) prepare_datetime: DataFrame ว่างเปล่า ({timeframe_str}). Returning empty DataFrame.")
        return df_pd

    try:
        if "Date" not in df_pd.columns or "Timestamp" not in df_pd.columns:
            logging.critical(f"(Error) ขาดคอลัมน์ 'Date'/'Timestamp' ใน {timeframe_str}.")
            sys.exit(f"ออก ({timeframe_str}): ขาดคอลัมน์ Date/Timestamp ที่จำเป็นสำหรับการเตรียม Datetime.")

        preview_datetime_format(df_pd)

        date_str_series = df_pd["Date"].astype(str).str.strip()
        ts_str_series = (
            df_pd["Timestamp"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

        logging.info("      [Converter] กำลังตรวจสอบและแปลงปี พ.ศ. เป็น ค.ศ. (ถ้าจำเป็น)...")
        converted_date_str_series = date_str_series.copy()
        if not date_str_series.empty:
            potential_be = False
            sample_size = min(len(date_str_series), 100)
            try:
                if date_str_series.index.is_unique:
                    sampled_dates = date_str_series.sample(sample_size, random_state=42)
                else:
                    sampled_dates = date_str_series.drop_duplicates().sample(min(sample_size, date_str_series.nunique()), random_state=42)
            except Exception as e_sample:
                logging.warning(f"      [Converter] Warning: Sampling failed ({e_sample}). Proceeding without sampling check.")
                sampled_dates = date_str_series

            for date_str_sample in sampled_dates:
                if isinstance(date_str_sample, str):
                    year_part_str = None
                    if len(date_str_sample) >= 4:
                        if date_str_sample[:4].isdigit(): year_part_str = date_str_sample[:4]
                    if year_part_str:
                        try:
                            year_part = int(year_part_str)
                            # Use datetime.datetime.now() with the globally imported datetime module
                            current_ce_year = datetime.datetime.now().year # <<< CORRECTED
                            if year_part > current_ce_year + 100:
                                potential_be = True
                                logging.debug(f"      [Converter] Potential BE year detected: {year_part} in '{date_str_sample}'")
                                break
                        except ValueError: continue
            del sampled_dates

            if potential_be:
                logging.info("      [Converter] ตรวจพบปีที่อาจเป็น พ.ศ. (> 2400). พยายามแปลงเป็น ค.ศ. (-543)...")
                def convert_be_year(date_str):
                    if isinstance(date_str, str) and len(date_str) >= 4:
                        year_part_str = date_str[:4]
                        if year_part_str.isdigit():
                            try:
                                year_be = int(year_part_str)
                                current_ce_year = datetime.datetime.now().year # <<< CORRECTED
                                if year_be > current_ce_year + 100:
                                    year_ce = year_be - 543
                                    return str(year_ce) + date_str[4:]
                            except ValueError:
                                return date_str
                    return date_str
                converted_date_str_series = date_str_series.apply(convert_be_year)
                if not converted_date_str_series.equals(date_str_series):
                    logging.info("      [Converter] (Success) แปลงปี พ.ศ. เป็น ค.ศ. สำเร็จ.")
                    diff_mask = date_str_series != converted_date_str_series
                    logging.info(f"         ตัวอย่างก่อนแปลง:\n{date_str_series[diff_mask].head(3).to_string(index=False)}")
                    logging.info(f"         ตัวอย่างหลังแปลง:\n{converted_date_str_series[diff_mask].head(3).to_string(index=False)}")
                    del diff_mask
                else:
                    logging.info("      [Converter] ไม่พบปีที่น่าจะเป็น พ.ศ. (หรือข้อมูลน้อยเกินไป).")
            else:
                logging.info("      [Converter] ไม่พบปีที่น่าจะเป็น พ.ศ. (หรือข้อมูลน้อยเกินไป).")

        logging.debug("      Combining Date and Timestamp strings...")
        datetime_combined_str = converted_date_str_series + " " + ts_str_series
        df_pd["datetime_original"] = parse_datetime_safely(datetime_combined_str)
        del date_str_series, ts_str_series, converted_date_str_series
        gc.collect()

        nat_count = df_pd["datetime_original"].isna().sum()
        if nat_count > 0:
            nat_ratio = nat_count / len(df_pd) if len(df_pd) > 0 else 0
            logging.warning(f"   (Warning) พบค่า NaT {nat_count} ({nat_ratio:.1%}) ใน {timeframe_str} หลังการ parse.")

            if nat_ratio == 1.0:
                failed_strings = datetime_combined_str[df_pd["datetime_original"].isna()]
                logging.critical(f"   (Error) พบค่า NaT 100% ใน {timeframe_str}. ไม่สามารถดำเนินการต่อได้. ตัวอย่าง: {failed_strings.iloc[0] if not failed_strings.empty else 'N/A'}")
                sys.exit(f"   ออก ({timeframe_str}): ข้อมูล date/time ทั้งหมดไม่สามารถ parse ได้.")
            elif nat_ratio >= MAX_NAT_RATIO_THRESHOLD:
                logging.warning(f"   (Warning) สัดส่วน NaT ({nat_ratio:.1%}) เกินเกณฑ์ ({MAX_NAT_RATIO_THRESHOLD:.1%}) แต่ไม่ใช่ 100%.")
                logging.warning(f"   (Warning) Fallback: ลบ {nat_count} แถว NaT และดำเนินการต่อ...")
                df_pd.dropna(subset=["datetime_original"], inplace=True)
                if df_pd.empty:
                    logging.critical(f"   (Error) ข้อมูล {timeframe_str} ทั้งหมดเป็น NaT หรือใช้ไม่ได้หลัง fallback (และ DataFrame ว่างเปล่า).")
                    sys.exit(f"   ออก ({timeframe_str}): ข้อมูลว่างเปล่าหลังลบ NaT เกินเกณฑ์.")
                logging.info(f"   (Success) ดำเนินการต่อด้วย {len(df_pd)} แถวที่เหลือ ({timeframe_str}).")
            else:
                logging.info(f"   กำลังลบ {nat_count} แถว NaT (ต่ำกว่าเกณฑ์).")
                df_pd.dropna(subset=["datetime_original"], inplace=True)
                if df_pd.empty:
                    logging.critical(f"   (Error) ข้อมูล {timeframe_str} ว่างเปล่าหลังลบ NaT จำนวนเล็กน้อย.")
                    sys.exit(f"   ออก ({timeframe_str}): ข้อมูลว่างเปล่าหลังลบ NaT.")
        else:
            logging.debug(f"   ไม่พบค่า NaT ใน {timeframe_str} หลังการ parse.")
        del datetime_combined_str
        gc.collect()

        if "datetime_original" in df_pd.columns:
            df_pd["datetime_original"] = pd.to_datetime(df_pd["datetime_original"], errors='coerce')
            df_pd = df_pd[~df_pd["datetime_original"].isna()]
            if df_pd.empty:
                logging.critical(f"   (Error) ข้อมูล {timeframe_str} ว่างเปล่าหลังแปลง datetime_original และลบ NaT (ก่อน set_index).")
                sys.exit(f"   ออก ({timeframe_str}): ข้อมูลว่างเปล่าหลังการเตรียม datetime.")
            df_pd.set_index(pd.DatetimeIndex(df_pd["datetime_original"]), inplace=True)
        else:
            logging.critical(f"   (Error) คอลัมน์ 'datetime_original' หายไปก่อนการตั้งค่า Index ({timeframe_str}).")
            sys.exit(f"   ออก ({timeframe_str}): ขาดคอลัมน์ 'datetime_original'.")

        df_pd.sort_index(inplace=True)

        if df_pd.index.has_duplicates:
            initial_rows_dedup = df_pd.shape[0]
            logging.warning(f"   (Warning) พบ Index ซ้ำ {df_pd.index.duplicated().sum()} รายการ. กำลังลบรายการซ้ำ (เก็บรายการแรก)...")
            df_pd = df_pd[~df_pd.index.duplicated(keep='first')]
            logging.info(f"   แก้ไข index ซ้ำ: ลบ {initial_rows_dedup - df_pd.shape[0]} แถว.")

        logging.debug("   Checking for non-monotonic index (time reversals)...")
        time_diffs = df_pd.index.to_series().diff()
        negative_diffs = time_diffs[time_diffs < pd.Timedelta(0)]
        if not negative_diffs.empty:
            logging.critical(f"   (CRITICAL WARNING) พบเวลาย้อนกลับใน Index ของ {timeframe_str} หลังการเรียงลำดับ!")
            logging.critical(f"      จำนวน: {len(negative_diffs)}")
            logging.critical(f"      ตัวอย่าง Index ที่มีปัญหา:\n{negative_diffs.head()}")
            sys.exit(f"   ออก ({timeframe_str}): พบเวลาย้อนกลับในข้อมูล.")
        else:
            logging.debug("      Index is monotonic increasing.")
        del time_diffs, negative_diffs

        logging.info(f"(Success) เตรียม Datetime index ({timeframe_str}) สำเร็จ. Shape: {df_pd.shape}")
        return df_pd

    except SystemExit as se:
        raise se
    except ValueError as ve:
        logging.critical(f"   (Error) prepare_datetime: ValueError: {ve}", exc_info=True)
        sys.exit(f"   ออก ({timeframe_str}): ปัญหาข้อมูล Date/time.")
    except Exception as e:
        logging.critical(f"(Error) ข้อผิดพลาดร้ายแรงใน prepare_datetime ({timeframe_str}): {e}", exc_info=True)
        sys.exit(f"   ออก ({timeframe_str}): ข้อผิดพลาดร้ายแรงในการเตรียม datetime.")

logging.info("Part 4: Data Loading & Initial Preparation Functions Loaded.")
# === END OF PART 4/12 ===
# === START OF PART 5/12 ===

# ==============================================================================
# === PART 5: Feature Engineering & Indicator Calculation (v4.8.4) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, enhanced NaN/error handling, fixed SyntaxError, added integer downcast >>>
# <<< Includes fixes from v4.7.8: Context column calculation, fixed UnboundLocalError, fixed TypeError >>>
# <<< MODIFIED v4.8.0: Ensured Trend_Zone is always category dtype on return >>>
# <<< MODIFIED v4.8.1: Added handling for empty series, NaN/Inf inputs, dtype checks in indicators; refined NaN filling in engineer_m1_features; improved dtype conversion in clean_m1_data >>>
# <<< MODIFIED v4.8.2: Ensured robust index conversion and handling in engineer_m1_features before get_session_tag >>>
# <<< MODIFIED v4.8.3: Refined session tagging in engineer_m1_features for non-DatetimeIndex and addressed FutureWarning. Re-indented and reviewed. Added Part Markers. >>>
# <<< MODIFIED v4.8.4: Corrected session tagging for non-DatetimeIndex in engineer_m1_features to align with test expectations. Updated versioning. >>>
import logging
import pandas as pd
import numpy as np
import ta # Assumes 'ta' is imported and available (checked in Part 1)
from sklearn.cluster import KMeans # For context column calculation
from sklearn.preprocessing import StandardScaler # For context column calculation
import gc # For memory management

# Ensure global configurations are accessible if run independently
DEFAULT_ROLLING_Z_WINDOW_M1 = 300; DEFAULT_ATR_ROLLING_AVG_PERIOD = 50
DEFAULT_PATTERN_BREAKOUT_Z_THRESH = 2.0; DEFAULT_PATTERN_REVERSAL_BODY_RATIO = 0.5
DEFAULT_PATTERN_STRONG_TREND_Z_THRESH = 1.0; DEFAULT_PATTERN_CHOPPY_CANDLE_RATIO = 0.3
DEFAULT_PATTERN_CHOPPY_WICK_RATIO = 0.6; DEFAULT_M15_TREND_EMA_FAST = 50
DEFAULT_M15_TREND_EMA_SLOW = 200; DEFAULT_M15_TREND_RSI_PERIOD = 14
DEFAULT_M15_TREND_RSI_UP = 52; DEFAULT_M15_TREND_RSI_DOWN = 48
DEFAULT_TIMEFRAME_MINUTES_M1 = 1; DEFAULT_MIN_SIGNAL_SCORE_ENTRY = 2.0
DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO = 1.8; DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO = 0.75
DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R = 0.5; DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R = 1.0
DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R = 0.3; DEFAULT_ADAPTIVE_TSL_START_ATR_MULT = 1.5

try: ROLLING_Z_WINDOW_M1
except NameError: ROLLING_Z_WINDOW_M1 = DEFAULT_ROLLING_Z_WINDOW_M1
try: ATR_ROLLING_AVG_PERIOD
except NameError: ATR_ROLLING_AVG_PERIOD = DEFAULT_ATR_ROLLING_AVG_PERIOD
try: PATTERN_BREAKOUT_Z_THRESH
except NameError: PATTERN_BREAKOUT_Z_THRESH = DEFAULT_PATTERN_BREAKOUT_Z_THRESH
try: PATTERN_REVERSAL_BODY_RATIO
except NameError: PATTERN_REVERSAL_BODY_RATIO = DEFAULT_PATTERN_REVERSAL_BODY_RATIO
try: PATTERN_STRONG_TREND_Z_THRESH
except NameError: PATTERN_STRONG_TREND_Z_THRESH = DEFAULT_PATTERN_STRONG_TREND_Z_THRESH
try: PATTERN_CHOPPY_CANDLE_RATIO
except NameError: PATTERN_CHOPPY_CANDLE_RATIO = DEFAULT_PATTERN_CHOPPY_CANDLE_RATIO
try: PATTERN_CHOPPY_WICK_RATIO
except NameError: PATTERN_CHOPPY_WICK_RATIO = DEFAULT_PATTERN_CHOPPY_WICK_RATIO
try: M15_TREND_EMA_FAST
except NameError: M15_TREND_EMA_FAST = DEFAULT_M15_TREND_EMA_FAST
try: M15_TREND_EMA_SLOW
except NameError: M15_TREND_EMA_SLOW = DEFAULT_M15_TREND_EMA_SLOW
try: M15_TREND_RSI_PERIOD
except NameError: M15_TREND_RSI_PERIOD = DEFAULT_M15_TREND_RSI_PERIOD
try: M15_TREND_RSI_UP
except NameError: M15_TREND_RSI_UP = DEFAULT_M15_TREND_RSI_UP
try: M15_TREND_RSI_DOWN
except NameError: M15_TREND_RSI_DOWN = DEFAULT_M15_TREND_RSI_DOWN
try: TIMEFRAME_MINUTES_M1
except NameError: TIMEFRAME_MINUTES_M1 = DEFAULT_TIMEFRAME_MINUTES_M1
try: MIN_SIGNAL_SCORE_ENTRY
except NameError: MIN_SIGNAL_SCORE_ENTRY = DEFAULT_MIN_SIGNAL_SCORE_ENTRY
try: ADAPTIVE_TSL_HIGH_VOL_RATIO
except NameError: ADAPTIVE_TSL_HIGH_VOL_RATIO = DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO
try: ADAPTIVE_TSL_LOW_VOL_RATIO
except NameError: ADAPTIVE_TSL_LOW_VOL_RATIO = DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO
try: ADAPTIVE_TSL_DEFAULT_STEP_R
except NameError: ADAPTIVE_TSL_DEFAULT_STEP_R = DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R
try: ADAPTIVE_TSL_HIGH_VOL_STEP_R
except NameError: ADAPTIVE_TSL_HIGH_VOL_STEP_R = DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R
try: ADAPTIVE_TSL_LOW_VOL_STEP_R
except NameError: ADAPTIVE_TSL_LOW_VOL_STEP_R = DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R
try: ADAPTIVE_TSL_START_ATR_MULT
except NameError: ADAPTIVE_TSL_START_ATR_MULT = DEFAULT_ADAPTIVE_TSL_START_ATR_MULT
try: META_CLASSIFIER_FEATURES
except NameError: META_CLASSIFIER_FEATURES = []
try: SESSION_TIMES_UTC
except NameError: SESSION_TIMES_UTC = {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)}


# --- Indicator Calculation Functions ---
def ema(series, period):
    if not isinstance(series, pd.Series): logging.error(f"EMA Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    if series.empty: logging.debug("EMA: Input series is empty, returning empty series."); return pd.Series(dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if series_numeric.empty: logging.warning("EMA: Series contains only NaN/Inf values or is empty after cleaning."); return pd.Series(np.nan, index=series.index, dtype='float32')
    try:
        ema_calculated = series_numeric.ewm(span=period, adjust=False, min_periods=max(1, period)).mean()
        ema_result = ema_calculated.reindex(series.index); del series_numeric, ema_calculated; gc.collect()
        return ema_result.astype('float32')
    except Exception as e: logging.error(f"EMA calculation failed for period {period}: {e}", exc_info=True); return pd.Series(np.nan, index=series.index, dtype='float32')

def sma(series, period):
    if not isinstance(series, pd.Series): logging.error(f"SMA Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    if series.empty: logging.debug("SMA: Input series is empty, returning empty series."); return pd.Series(dtype='float32')
    if not isinstance(period, int) or period <= 0: logging.error(f"SMA calculation failed: Invalid period ({period})."); return pd.Series(np.nan, index=series.index, dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    if series_numeric.isnull().all(): logging.warning("SMA: Series contains only NaN values after numeric conversion and fill."); return pd.Series(np.nan, index=series.index, dtype='float32')
    try:
        min_p = max(1, min(period, len(series_numeric)))
        sma_result = series_numeric.rolling(window=period, min_periods=min_p).mean()
        sma_final = sma_result.reindex(series.index); del series_numeric, sma_result; gc.collect()
        return sma_final.astype('float32')
    except Exception as e: logging.error(f"SMA calculation failed for period {period}: {e}", exc_info=True); return pd.Series(np.nan, index=series.index, dtype='float32')

def rsi(series, period=14):
    if not isinstance(series, pd.Series): logging.error(f"RSI Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    if series.empty: logging.debug("RSI: Input series is empty, returning empty series."); return pd.Series(dtype='float32')
    if 'ta' not in globals() or ta is None: logging.error("   (Error) RSI calculation failed: 'ta' library not loaded."); return pd.Series(np.nan, index=series.index, dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if series_numeric.empty or len(series_numeric) < period: logging.warning(f"   (Warning) RSI calculation skipped: Not enough valid data points ({len(series_numeric)} < {period})."); return pd.Series(np.nan, index=series.index, dtype='float32')
    try:
        rsi_indicator = ta.momentum.RSIIndicator(close=series_numeric, window=period, fillna=False)
        rsi_values = rsi_indicator.rsi(); rsi_final = rsi_values.reindex(series.index).ffill()
        del series_numeric, rsi_indicator, rsi_values; gc.collect()
        return rsi_final.astype('float32')
    except Exception as e: logging.error(f"   (Error) RSI calculation error for period {period}: {e}.", exc_info=True); return pd.Series(np.nan, index=series.index, dtype='float32')

def atr(df_in, period=14):
    if not isinstance(df_in, pd.DataFrame): logging.error(f"ATR Error: Input must be a pandas DataFrame, got {type(df_in)}"); raise TypeError("Input must be a pandas DataFrame.")
    atr_col_name = f"ATR_{period}"; atr_shifted_col_name = f"ATR_{period}_Shifted"
    if df_in.empty:
        df_result = df_in.copy(); df_result[atr_col_name] = np.nan; df_result[atr_shifted_col_name] = np.nan
        df_result[atr_col_name] = df_result[atr_col_name].astype('float32'); df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32'); return df_result
    df_temp = df_in.copy(); required_price_cols = ["High", "Low", "Close"]
    if not all(col in df_temp.columns for col in required_price_cols):
        logging.warning(f"   (Warning) ATR calculation skipped: Missing columns {required_price_cols}.")
        df_result = df_in.copy(); df_result[atr_col_name] = np.nan; df_result[atr_shifted_col_name] = np.nan
        df_result[atr_col_name] = df_result[atr_col_name].astype('float32'); df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32'); return df_result
    for col in required_price_cols: df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df_temp.dropna(subset=required_price_cols, inplace=True)
    if df_temp.empty or len(df_temp) < period:
        logging.warning(f"   (Warning) ATR calculation skipped: Not enough valid data (need >= {period}).")
        df_result = df_in.copy(); df_result[atr_col_name] = np.nan; df_result[atr_shifted_col_name] = np.nan
        df_result[atr_col_name] = df_result[atr_col_name].astype('float32'); df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32'); return df_result
    atr_series = None
    if 'ta' in globals() and ta is not None:
        try:
            atr_indicator = ta.volatility.AverageTrueRange(high=df_temp['High'], low=df_temp['Low'], close=df_temp['Close'], window=period, fillna=False)
            atr_series = atr_indicator.average_true_range(); del atr_indicator
        except Exception as e_ta_atr: logging.warning(f"   (Warning) TA library ATR calculation failed: {e_ta_atr}. Falling back."); atr_series = None
    if atr_series is None:
        try:
            df_temp['H-L'] = df_temp['High'] - df_temp['Low']; df_temp['H-PC'] = abs(df_temp['High'] - df_temp['Close'].shift(1)); df_temp['L-PC'] = abs(df_temp['Low'] - df_temp['Close'].shift(1))
            df_temp['TR'] = df_temp[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            if not df_temp.empty and len(df_temp) > 0:
                first_valid_index = df_temp.index[0]
                if first_valid_index in df_temp.index: df_temp.loc[first_valid_index, 'TR'] = df_temp.loc[first_valid_index, 'H-L']
            atr_series = df_temp['TR'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        except Exception as e_pd_atr:
            logging.error(f"   (Error) Pandas EWM ATR calculation failed: {e_pd_atr}", exc_info=True)
            df_result = df_in.copy(); df_result[atr_col_name] = np.nan; df_result[atr_shifted_col_name] = np.nan
            df_result[atr_col_name] = df_result[atr_col_name].astype('float32'); df_result[atr_shifted_col_name] = df_result[atr_shifted_col_name].astype('float32')
            del df_temp; gc.collect(); return df_result
    df_result = df_in.copy(); df_result[atr_col_name] = atr_series.reindex(df_in.index).astype('float32')
    df_result[atr_shifted_col_name] = atr_series.shift(1).reindex(df_in.index).astype('float32')
    del df_temp, atr_series; gc.collect(); return df_result

def macd(series, window_slow=26, window_fast=12, window_sign=9):
    if not isinstance(series, pd.Series): logging.error(f"MACD Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    if series.empty: nan_series = pd.Series(dtype='float32'); return nan_series, nan_series.copy(), nan_series.copy()
    nan_series_indexed = pd.Series(np.nan, index=series.index, dtype='float32')
    if len(series.dropna()) < window_slow: logging.debug(f"MACD: Input series too short after dropna ({len(series.dropna())} < {window_slow})."); return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()
    if 'ta' not in globals() or ta is None: logging.error("   (Error) MACD calculation failed: 'ta' library not loaded."); return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if series_numeric.empty or len(series_numeric) < window_slow: logging.warning(f"   (Warning) MACD calculation skipped: Not enough valid data points ({len(series_numeric)} < {window_slow})."); return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()
    try:
        macd_indicator = ta.trend.MACD(close=series_numeric, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign, fillna=False)
        macd_line_final = macd_indicator.macd().reindex(series.index).ffill().astype('float32')
        macd_signal_final = macd_indicator.macd_signal().reindex(series.index).ffill().astype('float32')
        macd_diff_final = macd_indicator.macd_diff().reindex(series.index).ffill().astype('float32')
        del series_numeric, macd_indicator; gc.collect()
        return (macd_line_final, macd_signal_final, macd_diff_final)
    except Exception as e: logging.error(f"   (Error) MACD calculation error: {e}.", exc_info=True); return nan_series_indexed, nan_series_indexed.copy(), nan_series_indexed.copy()

def rolling_zscore(series, window, min_periods=None):
    if not isinstance(series, pd.Series): logging.error(f"Rolling Z-Score Error: Input must be a pandas Series, got {type(series)}"); raise TypeError("Input must be a pandas Series.")
    if series.empty: logging.debug("Rolling Z-Score: Input series empty, returning empty series."); return pd.Series(dtype='float32')
    if len(series) < 2: logging.debug("Rolling Z-Score: Input series too short (< 2), returning zeros."); return pd.Series(0.0, index=series.index, dtype='float32')
    series_numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    if series_numeric.isnull().all(): logging.warning("Rolling Z-Score: Series contains only NaN values after numeric conversion and fill, returning zeros."); return pd.Series(0.0, index=series.index, dtype='float32')
    actual_window = min(window, len(series_numeric))
    if actual_window < 2: logging.debug(f"Rolling Z-Score: Adjusted window size ({actual_window}) < 2, returning zeros."); return pd.Series(0.0, index=series.index, dtype='float32')
    if min_periods is None: min_periods = max(2, min(10, int(actual_window * 0.1)))
    else: min_periods = max(2, min(min_periods, actual_window))
    try:
        rolling_mean = series_numeric.rolling(window=actual_window, min_periods=min_periods).mean()
        rolling_std = series_numeric.rolling(window=actual_window, min_periods=min_periods).std()
        with np.errstate(divide='ignore', invalid='ignore'): rolling_std_safe = rolling_std.replace(0, np.nan); z = (series_numeric - rolling_mean) / rolling_std_safe
        z_filled = z.fillna(0.0);
        if np.isinf(z_filled).any(): z_filled.replace([np.inf, -np.inf], 0.0, inplace=True)
        z_final = z_filled.reindex(series.index).fillna(0.0)
        del series_numeric, rolling_mean, rolling_std, rolling_std_safe, z, z_filled; gc.collect()
        return z_final.astype('float32')
    except Exception as e: logging.error(f"Rolling Z-Score calculation failed for window {window}: {e}", exc_info=True); return pd.Series(0.0, index=series.index, dtype='float32')

def tag_price_structure_patterns(df):
    logging.info("   (Processing) Tagging price structure patterns...")
    if not isinstance(df, pd.DataFrame): logging.error("Pattern Tagging Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df.empty: df["Pattern_Label"] = "Normal"; df["Pattern_Label"] = df["Pattern_Label"].astype('category'); return df
    required_cols = ["Gain_Z", "High", "Low", "Close", "Open", "MACD_hist", "Candle_Ratio", "Wick_Ratio", "Gain", "Candle_Body"]
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"      (Warning) Missing columns for Pattern Labeling. Setting all to 'Normal'.")
        df["Pattern_Label"] = "Normal"; df["Pattern_Label"] = df["Pattern_Label"].astype('category'); return df
    df_patterns = df.copy()
    for col in ["Gain_Z", "MACD_hist", "Candle_Ratio", "Wick_Ratio", "Gain", "Candle_Body"]: df_patterns[col] = pd.to_numeric(df_patterns[col], errors='coerce').fillna(0)
    for col in ["High", "Low", "Close", "Open"]:
        df_patterns[col] = pd.to_numeric(df_patterns[col], errors='coerce')
        if df_patterns[col].isnull().any(): df_patterns[col] = df_patterns[col].ffill().bfill()
    df_patterns["Pattern_Label"] = "Normal"
    prev_high = df_patterns["High"].shift(1); prev_low = df_patterns["Low"].shift(1); prev_gain = df_patterns["Gain"].shift(1).fillna(0); prev_body = df_patterns["Candle_Body"].shift(1).fillna(0); prev_macd_hist = df_patterns["MACD_hist"].shift(1).fillna(0)
    breakout_cond = ((df_patterns["Gain_Z"].abs() > PATTERN_BREAKOUT_Z_THRESH) | ((df_patterns["High"] > prev_high) & (df_patterns["Close"] > prev_high)) | ((df_patterns["Low"] < prev_low) & (df_patterns["Close"] < prev_low))).fillna(False)
    reversal_cond = (((prev_gain < 0) & (df_patterns["Gain"] > 0) & (df_patterns["Candle_Body"] > (prev_body * PATTERN_REVERSAL_BODY_RATIO))) | ((prev_gain > 0) & (df_patterns["Gain"] < 0) & (df_patterns["Candle_Body"] > (prev_body * PATTERN_REVERSAL_BODY_RATIO)))).fillna(False)
    inside_bar_cond = ((df_patterns["High"] < prev_high) & (df_patterns["Low"] > prev_low)).fillna(False)
    strong_trend_cond = (((df_patterns["Gain_Z"] > PATTERN_STRONG_TREND_Z_THRESH) & (df_patterns["MACD_hist"] > 0) & (df_patterns["MACD_hist"] > prev_macd_hist)) | ((df_patterns["Gain_Z"] < -PATTERN_STRONG_TREND_Z_THRESH) & (df_patterns["MACD_hist"] < 0) & (df_patterns["MACD_hist"] < prev_macd_hist))).fillna(False)
    choppy_cond = ((df_patterns["Candle_Ratio"] < PATTERN_CHOPPY_CANDLE_RATIO) & (df_patterns["Wick_Ratio"] > PATTERN_CHOPPY_WICK_RATIO)).fillna(False)
    df_patterns.loc[breakout_cond, "Pattern_Label"] = "Breakout"
    df_patterns.loc[reversal_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "Reversal"
    df_patterns.loc[inside_bar_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "InsideBar"
    df_patterns.loc[strong_trend_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "StrongTrend"
    df_patterns.loc[choppy_cond & (df_patterns["Pattern_Label"] == "Normal"), "Pattern_Label"] = "Choppy"
    logging.info(f"      Pattern Label Distribution:\n{df_patterns['Pattern_Label'].value_counts(normalize=True).round(3).to_string()}")
    df["Pattern_Label"] = df_patterns["Pattern_Label"].astype('category')
    del df_patterns, prev_high, prev_low, prev_gain, prev_body, prev_macd_hist, breakout_cond, reversal_cond, inside_bar_cond, strong_trend_cond, choppy_cond; gc.collect()
    return df

def calculate_m15_trend_zone(df_m15):
    logging.info("(Processing) กำลังคำนวณ M15 Trend Zone...")
    if not isinstance(df_m15, pd.DataFrame): logging.error("M15 Trend Zone Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df_m15.empty or "Close" not in df_m15.columns:
        result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"}); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category'); return result_df
    df = df_m15.copy()
    try:
        df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
        if df["Close"].isnull().all(): result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"}); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category'); return result_df
        df["EMA_Fast"] = ema(df["Close"], M15_TREND_EMA_FAST); df["EMA_Slow"] = ema(df["Close"], M15_TREND_EMA_SLOW); df["RSI"] = rsi(df["Close"], M15_TREND_RSI_PERIOD)
        df.dropna(subset=["EMA_Fast", "EMA_Slow", "RSI"], inplace=True)
        if df.empty: result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"}); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category'); return result_df
        is_up = (df["EMA_Fast"] > df["EMA_Slow"]) & (df["RSI"] > M15_TREND_RSI_UP); is_down = (df["EMA_Fast"] < df["EMA_Slow"]) & (df["RSI"] < M15_TREND_RSI_DOWN)
        df["Trend_Zone"] = "NEUTRAL"; df.loc[is_up, "Trend_Zone"] = "UP"; df.loc[is_down, "Trend_Zone"] = "DOWN"
        if not df.empty: logging.info(f"   การกระจาย M15 Trend Zone:\n{df['Trend_Zone'].value_counts(normalize=True).round(3).to_string()}")
        result_df = df[["Trend_Zone"]].reindex(df_m15.index).fillna("NEUTRAL"); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category')
        del df, is_up, is_down; gc.collect(); return result_df
    except Exception as e:
        logging.error(f"(Error) การคำนวณ M15 Trend Zone ล้มเหลว: {e}", exc_info=True)
        result_df = pd.DataFrame(index=df_m15.index, data={"Trend_Zone": "NEUTRAL"}); result_df["Trend_Zone"] = result_df["Trend_Zone"].astype('category'); return result_df

def get_session_tag(timestamp, session_times_utc=None):
    if session_times_utc is None:
        global SESSION_TIMES_UTC
        try: session_times_utc_local = SESSION_TIMES_UTC
        except NameError: logging.warning("get_session_tag: Global SESSION_TIMES_UTC not found, using default."); session_times_utc_local = {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)}
    else: session_times_utc_local = session_times_utc
    if pd.isna(timestamp): return "N/A"
    try:
        # Ensure timestamp is a pandas Timestamp object for tz_convert/tz_localize
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp) # Attempt conversion

        ts_utc = timestamp.tz_convert('UTC') if timestamp.tzinfo else timestamp.tz_localize('UTC')
        hour = ts_utc.hour; sessions = []
        for name, (start, end) in session_times_utc_local.items():
            if start <= end:
                if start <= hour < end: sessions.append(name)
            else:
                if hour >= start or hour < end: sessions.append(name)
        return "/".join(sorted(sessions)) if sessions else "Other"
    except Exception as e: logging.error(f"   (Error) Error in get_session_tag for {timestamp}: {e}", exc_info=True); return "Error_Tagging"

def engineer_m1_features(df_m1, timeframe_minutes=TIMEFRAME_MINUTES_M1, lag_features_config=None):
    logging.info("(Processing) กำลังสร้าง Features M1 (v4.8.4)...") # <<< MODIFIED v4.8.4
    if not isinstance(df_m1, pd.DataFrame): logging.error("Engineer M1 Features Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df_m1.empty: logging.warning("   (Warning) ข้ามการสร้าง Features M1: DataFrame ว่างเปล่า."); return df_m1
    df = df_m1.copy(); price_cols = ["Open", "High", "Low", "Close"]
    if any(col not in df.columns for col in price_cols):
        logging.warning(f"   (Warning) ขาดคอลัมน์ราคา M1. บาง Features อาจเป็น NaN.")
        base_feature_cols = ["Candle_Body", "Candle_Range", "Gain", "Candle_Ratio", "Upper_Wick", "Lower_Wick", "Wick_Length", "Wick_Ratio", "Gain_Z", "MACD_line", "MACD_signal", "MACD_hist", "MACD_hist_smooth", "ATR_14", "ATR_14_Shifted", "ATR_14_Rolling_Avg", "Candle_Speed", 'Volatility_Index', 'ADX', 'RSI']
        for col in base_feature_cols:
            if col not in df.columns: df[col] = np.nan
        if "Pattern_Label" not in df.columns: df["Pattern_Label"] = "N/A"
    else:
        for col in price_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=price_cols, inplace=True)
        if df.empty: logging.warning("   (Warning) M1 DataFrame ว่างเปล่าหลังลบราคา NaN."); return df
        df["Candle_Body"]=abs(df["Close"]-df["Open"]).astype('float32'); df["Candle_Range"]=(df["High"]-df["Low"]).astype('float32'); df["Gain"]=(df["Close"]-df["Open"]).astype('float32')
        df["Candle_Ratio"]=np.where(df["Candle_Range"].abs()>1e-9,df["Candle_Body"]/df["Candle_Range"],0.0).astype('float32'); df["Upper_Wick"]=(df["High"]-np.maximum(df["Open"],df["Close"])).astype('float32')
        df["Lower_Wick"]=(np.minimum(df["Open"],df["Close"])-df["Low"]).astype('float32'); df["Wick_Length"]=(df["Upper_Wick"]+df["Lower_Wick"]).astype('float32')
        df["Wick_Ratio"]=np.where(df["Candle_Range"].abs()>1e-9,df["Wick_Length"]/df["Candle_Range"],0.0).astype('float32'); df["Gain_Z"]=rolling_zscore(df["Gain"].fillna(0),window=ROLLING_Z_WINDOW_M1)
        df["MACD_line"],df["MACD_signal"],df["MACD_hist"]=macd(df["Close"])
        if "MACD_hist" in df.columns and df["MACD_hist"].notna().any(): df["MACD_hist_smooth"]=df["MACD_hist"].rolling(window=5,min_periods=1).mean().fillna(0).astype('float32')
        else: df["MACD_hist_smooth"]=np.nan; logging.warning("      (Warning) ไม่สามารถคำนวณ MACD_hist_smooth.")
        df=atr(df,14)
        if "ATR_14" in df.columns and df["ATR_14"].notna().any(): df["ATR_14_Rolling_Avg"]=sma(df["ATR_14"],ATR_ROLLING_AVG_PERIOD)
        else: df["ATR_14_Rolling_Avg"]=np.nan; logging.warning("      (Warning) ไม่สามารถคำนวณ ATR_14_Rolling_Avg.")
        df["Candle_Speed"]=(df["Gain"]/max(timeframe_minutes,1e-6)).astype('float32'); df["RSI"]=rsi(df["Close"],period=14)
    if lag_features_config and isinstance(lag_features_config,dict):
        for feature in lag_features_config.get('features',[]):
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                for lag in lag_features_config.get('lags',[]):
                    if isinstance(lag,int) and lag>0: df[f"{feature}_lag{lag}"]=df[feature].shift(lag).astype('float32')
    if 'ATR_14' in df.columns and 'ATR_14_Rolling_Avg' in df.columns and df['ATR_14_Rolling_Avg'].notna().any():
        df['Volatility_Index']=np.where(df['ATR_14_Rolling_Avg'].abs()>1e-9,df['ATR_14']/df['ATR_14_Rolling_Avg'],np.nan)
        df['Volatility_Index']=df['Volatility_Index'].ffill().fillna(1.0).astype('float32')
    else: df['Volatility_Index']=1.0; logging.warning("         (Warning) ไม่สามารถคำนวณ Volatility_Index.")
    if all(c in df.columns for c in ['High','Low','Close']) and ta:
        try:
            if len(df.dropna(subset=['High','Low','Close']))>=14*2+10: adx_indicator=ta.trend.ADXIndicator(df['High'],df['Low'],df['Close'],window=14,fillna=False); df['ADX']=adx_indicator.adx().ffill().fillna(25.0).astype('float32')
            else: df['ADX']=25.0; logging.warning("         (Warning) ไม่สามารถคำนวณ ADX: ข้อมูลไม่เพียงพอ.")
        except Exception as e_adx: df['ADX']=25.0; logging.warning(f"         (Warning) ไม่สามารถคำนวณ ADX: {e_adx}")
    else: df['ADX']=25.0
    if all(col in df.columns for col in ["Gain_Z","High","Low","Close","Open","MACD_hist","Candle_Ratio","Wick_Ratio","Gain","Candle_Body"]): df=tag_price_structure_patterns(df)
    else: df["Pattern_Label"]="N/A"; df["Pattern_Label"]=df["Pattern_Label"].astype('category'); logging.warning("   (Warning) ข้ามการ Tag Patterns.")
    if 'cluster' not in df.columns:
        try:
            cluster_features=['Gain_Z','Volatility_Index','Candle_Ratio','RSI','ADX']; features_present=[f for f in cluster_features if f in df.columns and df[f].notna().any()]
            if len(features_present)<2 or len(df[features_present].dropna())<3: df['cluster']=0; logging.warning("         (Warning) Not enough valid features/samples for clustering.")
            else:
                X_cluster_raw=df[features_present].copy().replace([np.inf,-np.inf],np.nan); X_cluster=X_cluster_raw.fillna(X_cluster_raw.median()).fillna(0)
                if len(X_cluster)>=3: scaler=StandardScaler(); X_scaled=scaler.fit_transform(X_cluster); kmeans=KMeans(n_clusters=3,random_state=42,n_init=10); df['cluster']=kmeans.fit_predict(X_scaled)
                else: df['cluster']=0; logging.warning("         (Warning) Not enough samples after cleaning for clustering.")
        except Exception as e_cluster: df['cluster']=0; logging.error(f"         (Error) Clustering failed: {e_cluster}.",exc_info=True)
        if 'cluster' in df.columns: df['cluster']=pd.to_numeric(df['cluster'],downcast='integer')
    if 'spike_score' not in df.columns:
        try:
            gain_z_abs=abs(pd.to_numeric(df.get('Gain_Z',0.0),errors='coerce').fillna(0.0)); wick_ratio=pd.to_numeric(df.get('Wick_Ratio',0.0),errors='coerce').fillna(0.0)
            atr_val=pd.to_numeric(df.get('ATR_14',1.0),errors='coerce').fillna(1.0).replace([np.inf,-np.inf],1.0)
            score=(wick_ratio*0.5+gain_z_abs*0.3+atr_val*0.2); score=np.where((atr_val>1.5)&(wick_ratio>0.6),score*1.2,score); df['spike_score']=score.clip(0,1).astype('float32')
        except Exception as e_spike: df['spike_score']=0.0; logging.error(f"         (Error) Spike score calculation failed: {e_spike}.",exc_info=True)
    if 'session' not in df.columns:
        logging.info("      Creating 'session' column...")
        try:
            # <<< START OF MODIFIED v4.8.4 LOGIC for session tagging >>>
            original_index_is_datetime = isinstance(df.index, pd.DatetimeIndex) and not df.index.hasnans

            if original_index_is_datetime:
                if not df.empty:
                    df['session'] = df.index.to_series().apply(get_session_tag)
                else:
                    df['session'] = "N/A_EmptyDF" # Should not happen if df_m1 is not empty
            else: # Original index is not a valid DatetimeIndex (e.g., RangeIndex)
                logging.warning("         (Warning) Original index is not a valid DatetimeIndex. Session tagging will result in 'Error_Tagging_Reindex_Fill'.")
                # For non-DatetimeIndex, the goal is to have 'Error_Tagging_Reindex_Fill'
                # This happens because reindexing a Series with DatetimeIndex (from get_session_tag)
                # back to a RangeIndex (or other non-DatetimeIndex) will likely result in NaNs
                # which are then filled.
                # We still attempt conversion to see if any part of it *could* be datetime.
                temp_index_for_apply = pd.to_datetime(df.index, errors='coerce')
                if not temp_index_for_apply.isna().all(): # If at least some conversion was possible
                    # Apply get_session_tag on the converted (potentially partial) DatetimeIndex
                    session_values_on_temp_index = temp_index_for_apply.to_series().apply(get_session_tag)
                    # Reindex these session values back to the original df's index.
                    # If original df.index was RangeIndex, this reindex will likely produce NaNs
                    # where the RangeIndex doesn't match the DatetimeIndex values from temp_index.
                    df['session'] = session_values_on_temp_index.reindex(df.index)
                    df['session'] = df['session'].fillna("Error_Tagging_Reindex_Fill")
                else: # All conversions to datetime failed
                    df['session'] = "Error_Index_Conv"
            # <<< END OF MODIFIED v4.8.4 LOGIC for session tagging >>>

            df['session'] = df['session'].astype('category')
            if not df.empty: logging.info(f"         Session distribution:\n{df['session'].value_counts(normalize=True).round(3).to_string()}")
        except Exception as e_session: logging.error(f"         (Error) Session calculation failed: {e_session}. Assigning 'Other'.", exc_info=True); df['session'] = "Other"; df['session'] = df['session'].astype('category')
    if 'model_tag' not in df.columns: df['model_tag'] = 'N/A'
    logging.info("(Success) สร้าง Features M1 (v4.8.4) เสร็จสิ้น.") # <<< MODIFIED v4.8.4
    return df.reindex(df_m1.index)

def clean_m1_data(df_m1):
    logging.info("(Processing) กำลังกำหนด Features M1 สำหรับ Drift และแปลงประเภท (v4.8.4)...") # <<< MODIFIED v4.8.4
    if not isinstance(df_m1, pd.DataFrame): logging.error("Clean M1 Data Error: Input must be a pandas DataFrame."); raise TypeError("Input must be a pandas DataFrame.")
    if df_m1.empty: logging.warning("   (Warning) ข้ามการทำความสะอาดข้อมูล M1: DataFrame ว่างเปล่า."); return pd.DataFrame(), []
    df_cleaned = df_m1.copy()
    potential_m1_features = ["Candle_Body", "Candle_Range", "Candle_Ratio", "Gain", "Gain_Z", "MACD_line", "MACD_signal", "MACD_hist", "MACD_hist_smooth", "ATR_14", "ATR_14_Shifted", "ATR_14_Rolling_Avg", "Candle_Speed", "Wick_Length", "Wick_Ratio", "Pattern_Label", "Signal_Score", 'Volatility_Index', 'ADX', 'RSI', 'cluster', 'spike_score', 'session']
    lag_cols_in_df = [col for col in df_cleaned.columns if '_lag' in col]
    potential_m1_features.extend(lag_cols_in_df)
    if META_CLASSIFIER_FEATURES: potential_m1_features.extend([f for f in META_CLASSIFIER_FEATURES if f not in potential_m1_features])
    potential_m1_features = sorted(list(dict.fromkeys(potential_m1_features)))
    m1_features_for_drift = [f for f in potential_m1_features if f in df_cleaned.columns]
    logging.info(f"   กำหนด {len(m1_features_for_drift)} Features M1 สำหรับ Drift: {m1_features_for_drift}")
    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        try:
            inf_mask = df_cleaned[numeric_cols].isin([np.inf, -np.inf])
            if inf_mask.any().any():
                cols_with_inf = df_cleaned[numeric_cols].columns[inf_mask.any()].tolist()
                logging.warning(f"      [Inf Check] พบ Inf ใน: {cols_with_inf}. กำลังแทนที่ด้วย NaN...")
                df_cleaned[cols_with_inf] = df_cleaned[cols_with_inf].replace([np.inf, -np.inf], np.nan)
            for col in numeric_cols: df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            cols_with_nan = df_cleaned[numeric_cols].columns[df_cleaned[numeric_cols].isnull().any()].tolist()
            if cols_with_nan:
                logging.info(f"      [NaN Check] พบ NaN ใน: {cols_with_nan}. กำลังเติมด้วย ffill().fillna(0)...")
                df_cleaned[cols_with_nan] = df_cleaned[cols_with_nan].ffill().fillna(0)
            for col in numeric_cols:
                if col not in df_cleaned.columns: continue
                if pd.api.types.is_integer_dtype(df_cleaned[col].dtype): df_cleaned[col] = pd.to_numeric(df_cleaned[col], downcast='integer')
                elif pd.api.types.is_float_dtype(df_cleaned[col].dtype) and df_cleaned[col].dtype != 'float32': df_cleaned[col] = df_cleaned[col].astype('float32')
        except Exception as e: logging.error(f"   (Error) เกิดข้อผิดพลาดในการแปลงประเภทข้อมูลหรือเติม NaN/Inf: {e}.", exc_info=True)
    categorical_cols = ['Pattern_Label', 'session']
    for col in categorical_cols:
        if col in df_cleaned.columns:
            if df_cleaned[col].isnull().any(): df_cleaned[col] = df_cleaned[col].fillna("Unknown") # Use assignment
            if not isinstance(df_cleaned[col].dtype, pd.CategoricalDtype):
                try: df_cleaned[col] = df_cleaned[col].astype('category')
                except Exception as e_cat: logging.warning(f"   (Warning) เกิดข้อผิดพลาดในการแปลง '{col}' เป็น category: {e_cat}.")
    logging.info("(Success) กำหนด Features M1 และแปลงประเภท (v4.8.4) เสร็จสิ้น.") # <<< MODIFIED v4.8.4
    return df_cleaned, m1_features_for_drift

def calculate_m1_entry_signals(df_m1: pd.DataFrame, config: dict) -> pd.DataFrame:
    logging.debug("      (Calculating M1 Signals)...")
    df = df_m1.copy(); df['Signal_Score'] = 0.0
    gain_z_thresh = config.get('gain_z_thresh', 0.3); rsi_thresh_buy = config.get('rsi_thresh_buy', 50)
    rsi_thresh_sell = config.get('rsi_thresh_sell', 50); volatility_max = config.get('volatility_max', 4.0)
    entry_score_min = config.get('min_signal_score', MIN_SIGNAL_SCORE_ENTRY); ignore_rsi = config.get('ignore_rsi_scoring', False)
    df['Gain_Z'] = df.get('Gain_Z', pd.Series(0.0, index=df.index)).fillna(0.0)
    buy_gain_z_cond = df['Gain_Z'] > gain_z_thresh; sell_gain_z_cond = df['Gain_Z'] < -gain_z_thresh
    df['Pattern_Label'] = df.get('Pattern_Label', pd.Series('Normal', index=df.index)).astype(str).fillna('Normal')
    buy_pattern_cond = df['Pattern_Label'].isin(['Breakout', 'StrongTrend']) & (df['Gain_Z'] > 0)
    sell_pattern_cond = df['Pattern_Label'].isin(['Breakout', 'StrongTrend', 'Reversal']) & (df['Gain_Z'] < 0)
    df['RSI'] = df.get('RSI', pd.Series(50.0, index=df.index)).fillna(50.0)
    buy_rsi_cond = df['RSI'] > rsi_thresh_buy; sell_rsi_cond = df['RSI'] < rsi_thresh_sell
    df['Volatility_Index'] = df.get('Volatility_Index', pd.Series(1.0, index=df.index)).fillna(1.0)
    vol_cond = df['Volatility_Index'] < volatility_max
    df.loc[buy_gain_z_cond, 'Signal_Score'] += 1.0; df.loc[sell_gain_z_cond, 'Signal_Score'] -= 1.0
    df.loc[buy_pattern_cond, 'Signal_Score'] += 1.0; df.loc[sell_pattern_cond, 'Signal_Score'] -= 1.0
    if not ignore_rsi: df.loc[buy_rsi_cond, 'Signal_Score'] += 1.0; df.loc[sell_rsi_cond, 'Signal_Score'] -= 1.0
    df.loc[vol_cond, 'Signal_Score'] += 1.0
    df['Signal_Score'] = df['Signal_Score'].astype('float32')
    df['Entry_Long'] = ((df['Signal_Score'] > 0) & (df['Signal_Score'] >= entry_score_min)).astype(int)
    df['Entry_Short'] = ((df['Signal_Score'] < 0) & (abs(df['Signal_Score']) >= entry_score_min)).astype(int)
    df['Trade_Reason'] = ""; df.loc[buy_gain_z_cond, 'Trade_Reason'] += f"+Gz>{gain_z_thresh:.1f}"
    df.loc[sell_gain_z_cond, 'Trade_Reason'] += f"+Gz<{-gain_z_thresh:.1f}"; df.loc[buy_pattern_cond, 'Trade_Reason'] += "+PBuy"
    df.loc[sell_pattern_cond, 'Trade_Reason'] += "+PSell"
    if not ignore_rsi: df.loc[buy_rsi_cond, 'Trade_Reason'] += f"+RSI>{rsi_thresh_buy}"; df.loc[sell_rsi_cond, 'Trade_Reason'] += f"+RSI<{rsi_thresh_sell}"
    df.loc[vol_cond, 'Trade_Reason'] += f"+Vol<{volatility_max:.1f}"
    buy_entry_mask = df['Entry_Long'] == 1; sell_entry_mask = df['Entry_Short'] == 1
    df.loc[buy_entry_mask, 'Trade_Reason'] = "BUY(" + df.loc[buy_entry_mask, 'Signal_Score'].round(1).astype(str) + "):" + df.loc[buy_entry_mask, 'Trade_Reason'].str.lstrip('+')
    df.loc[sell_entry_mask, 'Trade_Reason'] = "SELL(" + df.loc[sell_entry_mask, 'Signal_Score'].abs().round(1).astype(str) + "):" + df.loc[sell_entry_mask, 'Trade_Reason'].str.lstrip('+')
    df.loc[~(buy_entry_mask | sell_entry_mask), 'Trade_Reason'] = "NONE"
    df['Trade_Tag'] = df['Signal_Score'].round(1).astype(str) + "_" + df['Pattern_Label'].astype(str)
    return df

logging.info("Part 5: Feature Engineering & Indicator Calculation Functions Loaded.")
# === END OF PART 5/12 ===
# === START OF PART 6/12 ===

# ==============================================================================
# === PART 6: Machine Learning Configuration & Helpers (v4.8.8 - Patch 9 Applied: Fix ML Log) ===
# ==============================================================================
# <<< MODIFIED v4.8.7: Re-verified robustness checks in check_model_overfit based on latest prompt. >>>
# <<< MODIFIED v4.8.8: Ensured check_model_overfit robustness aligns with final prompt. Added default and try-except for META_META_MIN_PROBA_THRESH. >>>
# <<< MODIFIED v4.8.8 (Patch 1): Enhanced robustness in check_model_overfit, check_feature_noise_shap, analyze_feature_importance_shap, select_top_shap_features. Corrected logic for overfitting detection and noise logging. >>>
# <<< MODIFIED v4.8.8 (Patch 4): Corrected select_top_shap_features return value for invalid feature_names. Fixed overfitting detection logic and noise logging in check_model_overfit/check_feature_noise_shap. >>>
# <<< MODIFIED v4.8.8 (Patch 6): Applied user prompt fixes for check_model_overfit and check_feature_noise_shap logging. >>>
# <<< MODIFIED v4.8.8 (Patch 9): Fixed logging format and conditions in check_model_overfit and check_feature_noise_shap as per failed tests and plan. >>>
import logging
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
# Import ML libraries conditionally (assuming they are checked/installed in Part 1)
try:
    import shap
except ImportError:
    shap = None
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None
    Pool = None
try:
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
except ImportError:
    accuracy_score = roc_auc_score = log_loss = classification_report = lambda *args, **kwargs: None
    logging.error("Scikit-learn metrics not found!")

# Ensure global configurations are accessible if run independently
# Define defaults if globals are not found
DEFAULT_META_MIN_PROBA_THRESH = 0.55
DEFAULT_ENABLE_OPTUNA_TUNING = False
DEFAULT_OPTUNA_N_TRIALS = 50
DEFAULT_OPTUNA_CV_SPLITS = 5
DEFAULT_OPTUNA_METRIC = "AUC"
DEFAULT_OPTUNA_DIRECTION = "maximize"
DEFAULT_META_CLASSIFIER_FEATURES = [
    "RSI", "MACD_hist_smooth", "ATR_14", "ADX", "Gain_Z", "Volatility_Index",
    "Candle_Ratio", "Wick_Ratio", "Candle_Speed",
    "Gain_Z_lag1", "Gain_Z_lag3", "Gain_Z_lag5",
    "Candle_Speed_lag1", "Candle_Speed_lag3", "Candle_Speed_lag5",
    "cluster", "spike_score", "Pattern_Label",
]
# <<< [Patch] Added default for Meta-Meta threshold >>>
DEFAULT_META_META_MIN_PROBA_THRESH = 0.55

try:
    USE_META_CLASSIFIER
except NameError:
    USE_META_CLASSIFIER = True
try:
    USE_META_META_CLASSIFIER
except NameError:
    USE_META_META_CLASSIFIER = False
try:
    META_CLASSIFIER_PATH
except NameError:
    META_CLASSIFIER_PATH = "meta_classifier.pkl"
try:
    SPIKE_MODEL_PATH
except NameError:
    SPIKE_MODEL_PATH = "meta_classifier_spike.pkl"
try:
    CLUSTER_MODEL_PATH
except NameError:
    CLUSTER_MODEL_PATH = "meta_classifier_cluster.pkl"
try:
    META_META_CLASSIFIER_PATH
except NameError:
    META_META_CLASSIFIER_PATH = "meta_meta_classifier.pkl"
try:
    META_CLASSIFIER_FEATURES
except NameError:
    META_CLASSIFIER_FEATURES = DEFAULT_META_CLASSIFIER_FEATURES
try:
    META_META_CLASSIFIER_FEATURES
except NameError:
    META_META_CLASSIFIER_FEATURES = []
try:
    META_MIN_PROBA_THRESH
except NameError:
    META_MIN_PROBA_THRESH = DEFAULT_META_MIN_PROBA_THRESH
try:
    REENTRY_MIN_PROBA_THRESH
except NameError:
    REENTRY_MIN_PROBA_THRESH = META_MIN_PROBA_THRESH
# <<< [Patch] Added try-except for Meta-Meta threshold >>>
try:
    META_META_MIN_PROBA_THRESH
except NameError:
    META_META_MIN_PROBA_THRESH = DEFAULT_META_META_MIN_PROBA_THRESH
# <<< End of [Patch] >>>
try:
    ENABLE_OPTUNA_TUNING
except NameError:
    ENABLE_OPTUNA_TUNING = DEFAULT_ENABLE_OPTUNA_TUNING
try:
    OPTUNA_N_TRIALS
except NameError:
    OPTUNA_N_TRIALS = DEFAULT_OPTUNA_N_TRIALS
try:
    OPTUNA_CV_SPLITS
except NameError:
    OPTUNA_CV_SPLITS = DEFAULT_OPTUNA_CV_SPLITS
try:
    OPTUNA_METRIC
except NameError:
    OPTUNA_METRIC = DEFAULT_OPTUNA_METRIC
try:
    OPTUNA_DIRECTION
except NameError:
    OPTUNA_DIRECTION = DEFAULT_OPTUNA_DIRECTION

logging.info("Loading Machine Learning Configuration & Helpers...")

# --- ML Model Usage Flags ---
logging.info(f"USE_META_CLASSIFIER (L1 Filter): {USE_META_CLASSIFIER}")
logging.info(f"USE_META_META_CLASSIFIER (L2 Filter): {USE_META_META_CLASSIFIER}")

# --- ML Model Paths & Features ---
logging.debug(f"Main L1 Model Path: {META_CLASSIFIER_PATH}")
logging.debug(f"Spike Model Path: {SPIKE_MODEL_PATH}")
logging.debug(f"Cluster Model Path: {CLUSTER_MODEL_PATH}")
logging.debug(f"L2 Model Path: {META_META_CLASSIFIER_PATH}")
logging.debug(f"Default L1 Features (Count): {len(META_CLASSIFIER_FEATURES)}")
logging.debug(f"L2 Features (Count): {len(META_META_CLASSIFIER_FEATURES)}")

# --- ML Thresholds ---
logging.info(f"Default L1 Probability Threshold: {META_MIN_PROBA_THRESH}")
logging.info(f"Default Re-entry Probability Threshold: {REENTRY_MIN_PROBA_THRESH}")
logging.info(f"Default L2 Probability Threshold: {META_META_MIN_PROBA_THRESH}") # Now uses defined value

# --- Optuna Configuration ---
logging.info(f"Optuna Hyperparameter Tuning Enabled: {ENABLE_OPTUNA_TUNING}")
if ENABLE_OPTUNA_TUNING:
    logging.info(f"  Optuna Trials: {OPTUNA_N_TRIALS}")
    logging.info(f"  Optuna CV Splits: {OPTUNA_CV_SPLITS}")
    logging.info(f"  Optuna Metric: {OPTUNA_METRIC} ({OPTUNA_DIRECTION})")

# --- Auto Threshold Tuning ---
ENABLE_AUTO_THRESHOLD_TUNING = False # Keep this disabled for now
logging.info(f"Auto Threshold Tuning Enabled: {ENABLE_AUTO_THRESHOLD_TUNING}")

# --- Global variables to store model info ---
meta_model_type_used = "N/A"
meta_meta_model_type_used = "N/A"
logging.debug("Global model type trackers initialized.")

# --- SHAP Feature Selection Helper Function ---
def select_top_shap_features(shap_values_val, feature_names, shap_threshold=0.01):
    """
    Selects features based on Normalized Mean Absolute SHAP values exceeding a threshold.
    (v4.8.8 Patch 4: Corrected return for invalid feature_names)
    """
    logging.info(f"   [SHAP Select] กำลังเลือก Features ที่มี Normalized SHAP >= {shap_threshold:.4f}...")
    # <<< [Patch] MODIFIED v4.8.8 (Patch 1): Enhanced input validation >>>
    if shap_values_val is None or not isinstance(shap_values_val, np.ndarray) or shap_values_val.size == 0:
        logging.warning("      (Warning) ไม่สามารถเลือก Features: ค่า SHAP ไม่ถูกต้องหรือว่างเปล่า. คืนค่า Features เดิม.")
        return feature_names if isinstance(feature_names, list) else [] # Return original or empty list
    # <<< [Patch] MODIFIED v4.8.8 (Patch 4): Return None if feature_names is invalid >>>
    if feature_names is None or not isinstance(feature_names, list) or not feature_names:
        logging.warning("      (Warning) ไม่สามารถเลือก Features: รายชื่อ Features ไม่ถูกต้องหรือว่างเปล่า. คืนค่า None.")
        return None
    # <<< End of [Patch] MODIFIED v4.8.8 (Patch 4) >>>
    if shap_values_val.ndim != 2:
        # Handle potential case where SHAP returns values for multiple classes (e.g., list of arrays)
        if isinstance(shap_values_val, list) and len(shap_values_val) >= 2 and isinstance(shap_values_val[1], np.ndarray) and shap_values_val[1].ndim == 2:
            logging.debug("      (Info) SHAP values appear to be for multiple classes, using index 1 (positive class).")
            shap_values_val = shap_values_val[1] # Use SHAP values for the positive class
        else:
            logging.warning(f"      (Warning) ขนาด SHAP values ไม่ถูกต้อง ({shap_values_val.ndim} dimensions, expected 2). คืนค่า Features เดิม.")
            return feature_names
    if shap_values_val.shape[1] != len(feature_names):
        logging.warning(f"      (Warning) ขนาด SHAP values ไม่ตรงกับจำนวน Features (SHAP: {shap_values_val.shape[1]}, Features: {len(feature_names)}). คืนค่า Features เดิม.")
        return feature_names
    if shap_values_val.shape[0] == 0:
        logging.warning("      (Warning) SHAP values array มี 0 samples. ไม่สามารถคำนวณ Importance ได้. คืนค่า Features เดิม.")
        return feature_names
    # <<< End of [Patch] MODIFIED v4.8.8 (Patch 1) >>>

    try:
        mean_abs_shap = np.abs(shap_values_val).mean(axis=0)
        if np.isnan(mean_abs_shap).any() or np.isinf(mean_abs_shap).any():
            logging.warning("      (Warning) พบ NaN หรือ Inf ใน Mean Absolute SHAP values. ไม่สามารถเลือก Features ได้. คืนค่า Features เดิม.")
            return feature_names

        shap_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})
        total_shap = shap_df["Mean_Abs_SHAP"].sum()
        if total_shap > 1e-9:
            shap_df["Normalized_SHAP"] = shap_df["Mean_Abs_SHAP"] / total_shap
        else:
            shap_df["Normalized_SHAP"] = 0.0
            logging.warning("      (Warning) Total Mean Abs SHAP ใกล้ศูนย์, ไม่สามารถ Normalize ได้. จะไม่เลือก Feature ใดๆ.")
            return [] # Return empty list if no importance

        selected_features_df = shap_df[shap_df["Normalized_SHAP"] >= shap_threshold].copy()
        selected_features = selected_features_df["Feature"].tolist()

        if not selected_features:
            logging.warning(f"      (Warning) ไม่มี Features ใดผ่านเกณฑ์ SHAP >= {shap_threshold:.4f}. คืนค่า List ว่าง.")
            return []
        elif len(selected_features) < len(feature_names):
            removed_features = sorted(list(set(feature_names) - set(selected_features)))
            logging.info(f"      (Success) เลือก Features ได้ {len(selected_features)} ตัวจาก SHAP.")
            logging.info(f"         Features ที่ถูกตัดออก {len(removed_features)} ตัว: {removed_features}")
            logging.info("         Features ที่เลือก (เรียงตามค่า SHAP):")
            logging.info("\n" + selected_features_df.sort_values("Normalized_SHAP", ascending=False)[["Feature", "Normalized_SHAP"]].round(5).to_string(index=False))
        else:
            logging.info("      (Success) Features ทั้งหมดผ่านเกณฑ์ SHAP.")
        return selected_features
    except Exception as e:
        logging.error(f"      (Error) เกิดข้อผิดพลาดระหว่างการเลือก Features ด้วย SHAP: {e}. คืนค่า Features เดิม.", exc_info=True)
        return feature_names

# --- Model Quality Check Functions ---
def check_model_overfit(model, X_train, y_train, X_val, y_val, X_test=None, y_test=None, metric="AUC", threshold_pct=15.0):
    """
    Checks for potential overfitting by comparing model performance.
    (v4.8.8 Patch 9: Fixed logging logic and format)
    """
    logging.info(f"   [Check] Checking for Overfitting ({metric})...")
    if model is None:
        logging.warning("      (Warning) Cannot check Overfitting: Model is None.")
        return
    if X_val is None or y_val is None:
        logging.warning("      (Warning) Cannot check Overfitting: Validation data missing.")
        return
    if (X_test is None and y_test is not None) or (X_test is not None and y_test is None):
        logging.warning("      (Warning) Cannot check Overfitting: Test data requires both X_test and y_test.")
        X_test, y_test = None, None

    def _ensure_pd(data, name):
        if data is None: return None
        if isinstance(data, np.ndarray):
            if data.ndim == 1: return pd.Series(data, name=name)
            elif data.ndim == 2:
                feature_names = getattr(model, 'feature_names_', None)
                if feature_names and len(feature_names) == data.shape[1]:
                    return pd.DataFrame(data, columns=feature_names)
                else:
                    return pd.DataFrame(data)
            else:
                logging.warning(f"      (Warning) Cannot check Overfitting: {name} has unexpected dimensions ({data.ndim}).")
                return None
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return data
        else:
            logging.warning(f"      (Warning) Cannot check Overfitting: {name} has unexpected type ({type(data)}).")
            return None

    X_train = _ensure_pd(X_train, "X_train")
    y_train = _ensure_pd(y_train, "y_train")
    X_val = _ensure_pd(X_val, "X_val")
    y_val = _ensure_pd(y_val, "y_val")
    X_test = _ensure_pd(X_test, "X_test")
    y_test = _ensure_pd(y_test, "y_test")

    if X_val is None or y_val is None: return
    if X_train is not None and (y_train is None or len(X_train) != len(y_train)):
        logging.warning("      (Warning) Cannot check Overfitting: Train X and y data sizes do not match or y_train is None.")
        X_train, y_train = None, None
    if X_val is not None and (y_val is None or len(X_val) != len(y_val)):
        logging.warning("      (Warning) Cannot check Overfitting: Validation X and y data sizes do not match or y_val is None.")
        return
    if X_test is not None and (y_test is None or len(X_test) != len(y_test)):
        logging.warning("      (Warning) Cannot check Overfitting: Test X and y data sizes do not match or y_test is None.")
        X_test, y_test = None, None
    if X_val is not None and len(X_val) == 0:
        logging.warning("      (Warning) Cannot check Overfitting: Validation data is empty.")
        return
    if X_train is not None and len(X_train) == 0:
        logging.warning("      (Warning) Cannot check Overfitting: Train data is empty.")
        X_train, y_train = None, None
    if X_test is not None and len(X_test) == 0:
        logging.warning("      (Warning) Cannot check Overfitting: Test data is empty.")
        X_test, y_test = None, None

    try:
        train_score, val_score, test_score = np.nan, np.nan, np.nan
        model_classes = getattr(model, 'classes_', None)
        if model_classes is None or len(model_classes) < 2:
            logging.warning("      (Warning) Cannot determine model classes or only one class found. Using [0, 1] as fallback.")
            try:
                y_ref = y_val if y_train is None else y_train
                if y_ref is not None:
                    unique_classes = np.unique(y_ref)
                    if len(unique_classes) >= 2:
                        model_classes = unique_classes
                    else:
                        model_classes = [0, 1]
                else:
                    model_classes = [0, 1]
            except Exception:
                model_classes = [0, 1]

        def safe_predict(model, data, method_name):
            if data is None: return None
            if not hasattr(model, method_name):
                logging.warning(f"      (Warning) Model lacks '{method_name}' method.")
                return None
            try:
                pred = getattr(model, method_name)(data)
                if pred is None:
                    logging.warning(f"      (Warning) '{method_name}' returned None.")
                    return None
                if method_name == 'predict_proba':
                    if not isinstance(pred, np.ndarray) or pred.ndim != 2 or pred.shape[1] < 2:
                        logging.warning(f"      (Warning) '{method_name}' returned invalid shape or type: {getattr(pred, 'shape', type(pred))}.")
                        return None
                elif method_name == 'predict':
                    if not isinstance(pred, (np.ndarray, pd.Series)) or pred.ndim != 1:
                        logging.warning(f"      (Warning) '{method_name}' returned invalid shape or type: {getattr(pred, 'shape', type(pred))}.")
                        return None
                return pred
            except Exception as e_pred:
                logging.error(f"      (Error) Calculating {method_name} failed: {e_pred}", exc_info=True)
                return None

        if metric == "AUC":
            val_pred_proba_raw = safe_predict(model, X_val, "predict_proba")
            if val_pred_proba_raw is not None: val_score = roc_auc_score(y_val, val_pred_proba_raw[:, 1])
            train_pred_proba_raw = safe_predict(model, X_train, "predict_proba")
            if train_pred_proba_raw is not None: train_score = roc_auc_score(y_train, train_pred_proba_raw[:, 1])
            test_pred_proba_raw = safe_predict(model, X_test, "predict_proba")
            if test_pred_proba_raw is not None: test_score = roc_auc_score(y_test, test_pred_proba_raw[:, 1])
        elif metric == "LogLoss":
            val_pred_proba = safe_predict(model, X_val, "predict_proba")
            if val_pred_proba is not None: val_score = log_loss(y_val.astype(int), val_pred_proba, labels=model_classes)
            train_pred_proba = safe_predict(model, X_train, "predict_proba")
            if train_pred_proba is not None: train_score = log_loss(y_train.astype(int), train_pred_proba, labels=model_classes)
            test_pred_proba = safe_predict(model, X_test, "predict_proba")
            if test_pred_proba is not None: test_score = log_loss(y_test.astype(int), test_pred_proba, labels=model_classes)
        elif metric == "Accuracy":
            val_pred = safe_predict(model, X_val, "predict")
            if val_pred is not None: val_score = accuracy_score(y_val, val_pred)
            train_pred = safe_predict(model, X_train, "predict")
            if train_pred is not None: train_score = accuracy_score(y_train, train_pred)
            test_pred = safe_predict(model, X_test, "predict")
            if test_pred is not None: test_score = accuracy_score(y_test, test_pred)
        else:
            logging.warning(f"      (Warning) Metric '{metric}' not supported for Overfit check.")
            return

        if pd.notna(train_score): logging.info(f"      Train {metric}: {train_score:.4f}")
        if pd.notna(val_score): logging.info(f"      Val {metric}:   {val_score:.4f}")
        if pd.notna(test_score): logging.info(f"      Test {metric}:  {test_score:.4f}")

        # <<< [Patch] MODIFIED v4.8.8 (Patch 9): Corrected overfitting logic and logging >>>
        if pd.notna(train_score) and pd.notna(val_score):
            diff_val = train_score - val_score
            diff_val_pct = float('inf') # Default to infinity if denominator is zero
            is_overfitting_val = False
            denominator = 0.0

            if metric == "LogLoss": # Lower is better
                # Check if val_score is significantly worse (higher) than train_score
                denominator = abs(train_score)
                if val_score > train_score + 1e-9: # Check if val score is worse
                    if denominator > 1e-9:
                        diff_val_pct = abs(diff_val / denominator) * 100.0
                    if diff_val_pct > threshold_pct:
                        is_overfitting_val = True
            else: # Higher is better (AUC, Accuracy)
                # Check if train_score is significantly better (higher) than val_score
                denominator = abs(train_score)
                if train_score > val_score + 1e-9: # Check if train score is better
                    if denominator > 1e-9:
                        diff_val_pct = abs(diff_val / denominator) * 100.0
                    if diff_val_pct > threshold_pct:
                        is_overfitting_val = True

            # Log comparison results regardless of overfitting detection
            if denominator > 1e-9:
                logging.info(f"      Diff (Train - Val): {diff_val:.4f} ({diff_val_pct:.2f}%)")
            else:
                logging.info(f"      Diff (Train - Val): {diff_val:.4f} (Cannot calculate % diff - denominator near zero)")

            if is_overfitting_val:
                # Use the user-specified log message format
                logging.warning(f"[Patch] Potential Overfitting detected. Train vs Val {metric} gap = {abs(diff_val):.4f} ({diff_val_pct:.2f}% > {threshold_pct:.1f}%)")

        elif X_train is not None:
            logging.warning("      Diff (Train - Val): Cannot calculate (NaN score).")
        # <<< End of [Patch] MODIFIED v4.8.8 (Patch 9) >>>

        if pd.notna(val_score) and pd.notna(test_score):
            diff_test = val_score - test_score
            is_generalization_issue = False
            denominator_test = 0.0
            diff_test_pct = float('inf')

            if metric == "LogLoss": # Lower is better
                denominator_test = abs(val_score)
                if test_score > val_score + 1e-9: # Test score is worse
                    if denominator_test > 1e-9:
                        diff_test_pct = abs(diff_test / denominator_test) * 100.0
                    if diff_test_pct > threshold_pct:
                        is_generalization_issue = True
            else: # Higher is better
                denominator_test = abs(val_score)
                if val_score > test_score + 1e-9: # Val score is better
                    if denominator_test > 1e-9:
                        diff_test_pct = abs(diff_test / denominator_test) * 100.0
                    if diff_test_pct > threshold_pct:
                        is_generalization_issue = True

            if denominator_test > 1e-9:
                 logging.info(f"      Diff (Val - Test): {diff_test:.4f} ({diff_test_pct:.2f}%)")
            else:
                 logging.info(f"      Diff (Val - Test): {diff_test:.4f} (Cannot calculate % diff - denominator near zero)")

            if is_generalization_issue:
                logging.warning(f"      (ALERT) Potential Generalization Issue: Val {metric} significantly {'lower' if metric=='LogLoss' else 'higher'} than Test {metric} (Diff % > {threshold_pct:.1f}%).")
        elif X_test is not None:
            logging.warning("      Diff (Val - Test): Cannot calculate (NaN score).")

    except Exception as e:
        logging.error(f"      (Error) Error during Overfitting check ({metric}): {e}", exc_info=True)

def check_feature_noise_shap(shap_values, feature_names, threshold=0.01):
    """
    Checks for potentially noisy features based on low mean absolute SHAP values.
    (v4.8.8 Patch 9: Fixed logging logic and format)
    """
    logging.info("   [Check] Checking for Feature Noise (SHAP)...")
    if shap_values is None or not isinstance(shap_values, np.ndarray) or not feature_names or not isinstance(feature_names, list) or \
       shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names) or shap_values.shape[0] == 0:
        logging.warning("      (Warning) Skipping Feature Noise Check: Invalid inputs."); return

    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        if np.isnan(mean_abs_shap).any() or np.isinf(mean_abs_shap).any():
            logging.warning("      (Warning) Found NaN/Inf in Mean Abs SHAP. Skipping noise check.")
            return

        shap_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})
        total_shap = shap_df["Mean_Abs_SHAP"].sum()
        shap_df["Normalized_SHAP"] = (shap_df["Mean_Abs_SHAP"] / total_shap) if total_shap > 1e-9 else 0.0

        # <<< [Patch] MODIFIED v4.8.8 (Patch 9): Corrected logging as per user prompt and test failure >>>
        # Use the DataFrame index directly if feature_names match the index
        shap_series_for_check = pd.Series(shap_df["Normalized_SHAP"].values, index=shap_df["Feature"])
        noise_feats = shap_series_for_check[shap_series_for_check < threshold].index.tolist()
        if noise_feats:
            # Use the user-specified log message format (logging.info as per patch 6 note)
            logging.info(f"[Patch] SHAP Noise features detected: {noise_feats}")
        # <<< End of [Patch] MODIFIED v4.8.8 (Patch 9) >>>
        else:
            logging.info(f"      (Success) No features with significant noise detected (Normalized SHAP < {threshold:.4f}).")
    except Exception as e:
        logging.error(f"      (Error) Error during Feature Noise check (SHAP): {e}", exc_info=True)

# --- SHAP Analysis Function ---
def analyze_feature_importance_shap(model, model_type, data_sample, features, output_dir, fold_idx=None):
    """
    Analyzes feature importance using SHAP values and saves summary plots.
    (v4.8.8 Patch 1: Enhanced robustness for SHAP value structure and feature validation)
    """
    global shap
    if not shap:
        logging.warning("   (Warning) Skipping SHAP: 'shap' library not found.")
        return
    if model is None:
        logging.warning("   (Warning) Skipping SHAP: Model is None.")
        return
    if data_sample is None or not isinstance(data_sample, pd.DataFrame) or data_sample.empty:
        logging.warning("   (Warning) Skipping SHAP: No sample data.")
        return
    if not features or not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        logging.warning("   (Warning) Skipping SHAP: Invalid features list.")
        return
    if not output_dir or not os.path.isdir(output_dir):
        logging.warning(f"   (Warning) Skipping SHAP: Output directory '{output_dir}' invalid.")
        return

    fold_suffix = f"_fold{fold_idx+1}" if fold_idx is not None else "_validation_set"
    logging.info(f"\n(Analyzing) SHAP analysis ({model_type} - Sample Size: {len(data_sample)}) - {fold_suffix.replace('_',' ').title()}...")

    missing_features = [f for f in features if f not in data_sample.columns]
    if missing_features:
        logging.error(f"   (Error) Skipping SHAP: Missing features in data_sample: {missing_features}")
        return
    try:
        X_shap = data_sample[features].copy()
    except KeyError as e:
        logging.error(f"   (Error) Skipping SHAP: Feature(s) not found: {e}")
        return
    except Exception as e_select:
        logging.error(f"   (Error) Skipping SHAP: Error selecting features: {e_select}", exc_info=True)
        return

    cat_features_indices = []
    cat_feature_names_shap = []
    potential_cat_cols = ['Pattern_Label', 'session', 'Trend_Zone']
    logging.debug("      Processing categorical features for SHAP...")
    for cat_col in potential_cat_cols:
        if cat_col in X_shap.columns:
            try:
                if X_shap[cat_col].isnull().any():
                    X_shap[cat_col].fillna("Missing", inplace=True)
                X_shap[cat_col] = X_shap[cat_col].astype(str)
                if model_type == "CatBoostClassifier":
                    cat_feature_names_shap.append(cat_col)
            except Exception as e_cat_str:
                logging.warning(f"      (Warning) Could not convert '{cat_col}' to string for SHAP: {e_cat_str}.")

    if model_type == "CatBoostClassifier" and cat_feature_names_shap:
        try:
            cat_features_indices = [X_shap.columns.get_loc(col) for col in cat_feature_names_shap]
            logging.debug(f"         Categorical Feature Indices for SHAP Pool: {cat_features_indices}")
        except KeyError as e_loc:
            logging.error(f"      (Error) Could not locate categorical feature index for SHAP: {e_loc}.")
            cat_features_indices = []

    logging.debug("      Handling NaN/Inf in numeric features for SHAP...")
    numeric_cols_shap = X_shap.select_dtypes(include=np.number).columns
    if X_shap[numeric_cols_shap].isin([np.inf, -np.inf]).any().any():
        X_shap[numeric_cols_shap] = X_shap[numeric_cols_shap].replace([np.inf, -np.inf], np.nan)
    if X_shap[numeric_cols_shap].isnull().any().any():
        X_shap[numeric_cols_shap] = X_shap[numeric_cols_shap].fillna(0)

    if X_shap.isnull().any().any():
        missing_final = X_shap.columns[X_shap.isnull().any()].tolist()
        logging.error(f"      (Error) Skipping SHAP: NaNs still present after fill in columns: {missing_final}")
        return

    try:
        explainer = None
        shap_values = None
        global CatBoostClassifier, Pool

        logging.debug(f"      Initializing SHAP explainer for model type: {model_type}...")
        if model_type == "CatBoostClassifier" and CatBoostClassifier and Pool:
            shap_pool = Pool(X_shap, label=None, cat_features=cat_features_indices)
            explainer = shap.TreeExplainer(model)
            logging.info(f"      Calculating SHAP values (CatBoost)...")
            shap_values = explainer.shap_values(shap_pool)
        else:
            logging.warning(f"      (Warning) SHAP explainer not supported or library missing for model type: {model_type}")
            return

        shap_values_positive_class = None
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            if isinstance(shap_values[1], np.ndarray) and shap_values[1].ndim == 2:
                shap_values_positive_class = shap_values[1]
            else:
                logging.error(f"      (Error) SHAP values list element 1 has unexpected type/shape: {type(shap_values[1])}, {getattr(shap_values[1], 'shape', 'N/A')}")
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            shap_values_positive_class = shap_values
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            if shap_values.shape[0] >= 2 and shap_values.shape[1] == X_shap.shape[0] and shap_values.shape[2] == X_shap.shape[1]:
                shap_values_positive_class = shap_values[1, :, :]
            elif shap_values.shape[2] >= 2 and shap_values.shape[0] == X_shap.shape[0] and shap_values.shape[1] == X_shap.shape[1]:
                shap_values_positive_class = shap_values[:, :, 1]
            elif shap_values.shape[0] == 1:
                shap_values_positive_class = shap_values[0, :, :]
                logging.warning("      SHAP values have only one class output, using index 0.")
            else:
                logging.error(f"      (Error) Unexpected 3D SHAP values shape: {shap_values.shape}. Cannot determine positive class.")
        else:
            logging.error(f"      (Error) Unexpected SHAP values structure (Type: {type(shap_values)}, Shape: {getattr(shap_values, 'shape', 'N/A')}). Cannot plot.")
            return

        if shap_values_positive_class is None:
            logging.error("      (Error) Could not identify SHAP values for positive class.")
            return
        if shap_values_positive_class.shape[1] != len(features):
            logging.error(f"      (Error) SHAP feature dimension mismatch ({shap_values_positive_class.shape[1]} != {len(features)}). Cannot proceed.")
            return
        if shap_values_positive_class.shape[0] != X_shap.shape[0]:
            logging.error(f"      (Error) SHAP sample dimension mismatch ({shap_values_positive_class.shape[0]} != {X_shap.shape[0]}). Cannot proceed.")
            return

        logging.info("      Creating SHAP Summary Plot (bar type)...")
        shap_plot_path = os.path.join(output_dir, f"shap_summary_{model_type}_bar{fold_suffix}.png")
        plt.figure()
        try:
            shap.summary_plot(shap_values_positive_class, X_shap, plot_type="bar", show=False, feature_names=features, max_display=20)
            plt.title(f"SHAP Feature Importance ({model_type} - {fold_suffix.replace('_',' ').title()})")
            plt.tight_layout()
            plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
            logging.info(f"      (Success) Saved SHAP Plot (Bar): {os.path.basename(shap_plot_path)}")
        except Exception as e_save_bar:
            logging.error(f"      (Error) Failed to create/save SHAP bar plot: {e_save_bar}", exc_info=True)
        finally:
            plt.close()

        logging.info("      Creating SHAP Summary Plot (beeswarm/dot type)...")
        shap_beeswarm_path = os.path.join(output_dir, f"shap_summary_{model_type}_beeswarm{fold_suffix}.png")
        plt.figure()
        try:
            shap.summary_plot(shap_values_positive_class, X_shap, plot_type="dot", show=False, feature_names=features, max_display=20)
            plt.title(f"SHAP Feature Summary ({model_type} - {fold_suffix.replace('_',' ').title()})")
            plt.tight_layout()
            plt.savefig(shap_beeswarm_path, dpi=300, bbox_inches="tight")
            logging.info(f"      (Success) Saved SHAP Plot (Beeswarm): {os.path.basename(shap_beeswarm_path)}")
        except Exception as e_save_beeswarm:
            logging.error(f"      (Error) Failed to create/save SHAP beeswarm plot: {e_save_beeswarm}", exc_info=True)
        finally:
            plt.close()

    except ImportError:
        logging.error("   (Error) SHAP Error: Could not import shap library components.")
    except Exception as e:
        logging.error(f"   (Error) Error during SHAP analysis: {e}", exc_info=True)

# --- Feature Loading Function ---
def load_features_for_model(model_name, output_dir):
    """
    Loads the feature list for a specific model purpose from a JSON file.
    Falls back to loading 'features_main.json' if the specific file is not found.
    """
    features_filename = f"features_{model_name}.json"
    features_file_path = os.path.join(output_dir, features_filename)
    logging.info(f"   (Feature Load) Attempting to load features for '{model_name}' from: {features_file_path}")

    if not os.path.exists(features_file_path):
        logging.warning(f"   (Warning) Feature file not found for model '{model_name}': {os.path.basename(features_file_path)}")
        main_features_path = os.path.join(output_dir, "features_main.json")
        if model_name != 'main' and os.path.exists(main_features_path):
            logging.info(f"      (Fallback) Loading features from 'features_main.json' instead.")
            features_file_path = main_features_path # Use main path for fallback
        else:
            logging.error(f"      (Fallback Failed) Main feature file also not found or was requested.")
            return None # Return None if neither specific nor main exists

    try:
        with open(features_file_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        if isinstance(features, list) and all(isinstance(feat, str) for feat in features):
            logging.info(f"      (Success) Loaded {len(features)} features for model '{model_name}' from '{os.path.basename(features_file_path)}'.")
            return features
        else:
            logging.error(f"   (Error) Invalid format in feature file: {features_file_path}. Expected list of strings.")
            return None
    except json.JSONDecodeError as e_json:
        logging.error(f"   (Error) Failed to decode JSON from feature file '{os.path.basename(features_file_path)}': {e_json}")
        return None
    except Exception as e:
        logging.error(f"   (Error) Failed to load features for model '{model_name}' from '{os.path.basename(features_file_path)}': {e}", exc_info=True)
        return None

# --- Model Switcher Function ---
def select_model_for_trade(context, available_models=None):
    """
    Selects the appropriate AI model ('main', 'spike', 'cluster') based on context.
    Falls back to 'main' if the selected model is invalid or missing.
    """
    selected_model_key = 'main' # Default model
    confidence = None

    cluster_value = context.get('cluster')
    spike_score_value = context.get('spike_score', 0.0)

    if not isinstance(cluster_value, (int, float, np.number)) or pd.isna(cluster_value):
        cluster_value = None
    if not isinstance(spike_score_value, (int, float, np.number)) or pd.isna(spike_score_value):
        spike_score_value = 0.0

    spike_switch_threshold = 0.6
    cluster_switch_value = 2

    logging.debug(f"      (Switcher) Context: SpikeScore={spike_score_value:.3f}, Cluster={cluster_value}")

    if spike_score_value > spike_switch_threshold:
        selected_model_key = 'spike'
        confidence = spike_score_value
    elif cluster_value == cluster_switch_value:
        selected_model_key = 'cluster'
        confidence = 0.8
    else:
        selected_model_key = 'main'
        confidence = None

    if available_models is None:
        logging.error("      (Switcher Error) 'available_models' is None. Defaulting to 'main'.")
        selected_model_key = 'main'
        confidence = None
    elif selected_model_key not in available_models or \
         available_models.get(selected_model_key, {}).get('model') is None or \
         not available_models.get(selected_model_key, {}).get('features'):
        logging.warning(f"      (Switcher Warning) Selected model '{selected_model_key}' invalid/missing. Defaulting to 'main'.")
        selected_model_key = 'main'
        confidence = None

    logging.debug(f"      (Switcher) Final Selected Model: '{selected_model_key}', Confidence: {confidence}")
    return selected_model_key, confidence

logging.info("Part 6: Machine Learning Configuration & Helpers Loaded (v4.8.8 Patch 9 Applied).")
# === END OF PART 6/12 ===
# === START OF PART 7/12 ===

# ==============================================================================
# === PART 7: Model Training Function (v4.8.8 - Patch 2 Applied) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, enhanced error handling & GPU usage, fixed SyntaxError, added memory cleanup >>>
# <<< MODIFIED v4.8.1: Ensured GPU settings, early stopping, RAM optimization (sample_size, float32), and categorical handling are correctly implemented >>>
# <<< MODIFIED v4.8.2: Added robust input validation for trade_log_df and m1_df in train_and_export_meta_model, and updated log messages >>>
# <<< MODIFIED v4.8.8 (Patch 2): Fixed UnboundLocalError for y_val_cat by moving evaluation inside try block. Ensured Pool object is used correctly for CatBoost fit/eval. Added robustness checks. >>>
import logging
import os
import time
import json
import pandas as pd
import numpy as np
import traceback
from joblib import dump as joblib_dump # Use joblib dump directly
from sklearn.model_selection import train_test_split, TimeSeriesSplit # Ensure TimeSeriesSplit is imported
import gc # For memory management
# Import ML libraries conditionally (assuming they are checked/installed in Part 1)
try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    CatBoostClassifier = None
    Pool = None
try:
    import shap
except ImportError:
    shap = None
try:
    import optuna
except ImportError:
    optuna = None

# Ensure global configurations are accessible if run independently
# Define defaults if globals are not found
DEFAULT_META_CLASSIFIER_PATH = "meta_classifier.pkl"
DEFAULT_SPIKE_MODEL_PATH = "meta_classifier_spike.pkl"
DEFAULT_CLUSTER_MODEL_PATH = "meta_classifier_cluster.pkl"
DEFAULT_MODEL_TO_LINK = "catboost"
DEFAULT_ENABLE_OPTUNA_TUNING = False
DEFAULT_OPTUNA_N_TRIALS = 50
DEFAULT_OPTUNA_CV_SPLITS = 5
DEFAULT_OPTUNA_METRIC = "AUC"
DEFAULT_OPTUNA_DIRECTION = "maximize"
DEFAULT_META_CLASSIFIER_FEATURES = [] # Should be loaded or defined globally
DEFAULT_SHAP_IMPORTANCE_THRESHOLD = 0.01
DEFAULT_PERMUTATION_IMPORTANCE_THRESHOLD = 0.001
DEFAULT_CATBOOST_GPU_RAM_PART = 0.95 # Default from prompt
DEFAULT_SAMPLE_SIZE = 60000
DEFAULT_FEATURES_TO_DROP = None
DEFAULT_EARLY_STOPPING_ROUNDS = 200 # Default early stopping

try:
    META_CLASSIFIER_PATH
except NameError:
    META_CLASSIFIER_PATH = DEFAULT_META_CLASSIFIER_PATH
try:
    SPIKE_MODEL_PATH
except NameError:
    SPIKE_MODEL_PATH = DEFAULT_SPIKE_MODEL_PATH
try:
    CLUSTER_MODEL_PATH
except NameError:
    CLUSTER_MODEL_PATH = DEFAULT_CLUSTER_MODEL_PATH
try:
    DEFAULT_MODEL_TO_LINK
except NameError:
    DEFAULT_MODEL_TO_LINK = DEFAULT_MODEL_TO_LINK
try:
    ENABLE_OPTUNA_TUNING
except NameError:
    ENABLE_OPTUNA_TUNING = DEFAULT_ENABLE_OPTUNA_TUNING
try:
    OPTUNA_N_TRIALS
except NameError:
    OPTUNA_N_TRIALS = DEFAULT_OPTUNA_N_TRIALS
try:
    OPTUNA_CV_SPLITS
except NameError:
    OPTUNA_CV_SPLITS = DEFAULT_OPTUNA_CV_SPLITS
try:
    OPTUNA_METRIC
except NameError:
    OPTUNA_METRIC = DEFAULT_OPTUNA_METRIC
try:
    OPTUNA_DIRECTION
except NameError:
    OPTUNA_DIRECTION = DEFAULT_OPTUNA_DIRECTION
try:
    META_CLASSIFIER_FEATURES
except NameError:
    META_CLASSIFIER_FEATURES = DEFAULT_META_CLASSIFIER_FEATURES
try:
    shap_importance_threshold
except NameError:
    shap_importance_threshold = DEFAULT_SHAP_IMPORTANCE_THRESHOLD
try:
    permutation_importance_threshold
except NameError:
    permutation_importance_threshold = DEFAULT_PERMUTATION_IMPORTANCE_THRESHOLD
try:
    catboost_gpu_ram_part
except NameError:
    catboost_gpu_ram_part = DEFAULT_CATBOOST_GPU_RAM_PART
try:
    sample_size
except NameError:
    sample_size = DEFAULT_SAMPLE_SIZE
try:
    features_to_drop
except NameError:
    features_to_drop = DEFAULT_FEATURES_TO_DROP
try:
    early_stopping_rounds_config
except NameError:
    early_stopping_rounds_config = DEFAULT_EARLY_STOPPING_ROUNDS

# --- Meta Model Training Function ---
def train_and_export_meta_model(
    trade_log_path="trade_log_v32_walkforward.csv",
    m1_data_path="final_data_m1_v32_walkforward.csv",
    output_dir=None,
    model_purpose='main',
    trade_log_df_override=None,
    model_type_to_train="catboost",
    link_model_as_default="catboost",
    enable_dynamic_feature_selection=True,
    feature_selection_method='shap',
    shap_importance_threshold=shap_importance_threshold,
    permutation_importance_threshold=permutation_importance_threshold,
    prelim_model_params=None,
    enable_optuna_tuning=ENABLE_OPTUNA_TUNING,
    optuna_n_trials=OPTUNA_N_TRIALS,
    optuna_cv_splits=OPTUNA_CV_SPLITS,
    optuna_metric=OPTUNA_METRIC,
    optuna_direction=OPTUNA_DIRECTION,
    drift_observer=None,
    catboost_gpu_ram_part=catboost_gpu_ram_part,
    optuna_n_jobs=-1,
    sample_size=sample_size,
    features_to_drop_before_train=features_to_drop,
    early_stopping_rounds=early_stopping_rounds_config,
):
    """
    Trains and exports a Meta Classifier (L1) model for a specific purpose
    (main, spike, cluster) using trade log data and M1 features. Includes options
    for dynamic feature selection and hyperparameter optimization (Optuna).
    (v4.8.8 Patch 2: Fixed UnboundLocalError, Pool usage)

    Args:
        # ... (Args remain the same) ...

    Returns:
        tuple[dict, list]: A tuple containing:
            - saved_model_paths (dict): Dictionary mapping model purpose to the saved model file path.
                                        Returns None if training fails critically.
            - final_features_used (list): List of feature names used for the final trained model.
                                          Returns empty list if training fails.
    """
    start_train_time = time.time()
    logging.info(f"\n(Training - v4.8.8 Patch 2) เริ่มต้นการ Train Meta Classifier (Purpose: {model_purpose.upper()})...") # Updated version in log
    logging.info(f"   Model Type: {model_type_to_train}")
    logging.info(f"   Sample Size Limit: {sample_size}")
    logging.info(f"   Features to Drop Before Final Train: {features_to_drop_before_train}")
    logging.info(f"   Dynamic Feature Selection: {'เปิดใช้งาน' if enable_dynamic_feature_selection else 'ปิดใช้งาน'}")
    if enable_dynamic_feature_selection:
        logging.info(f"     Method: {feature_selection_method.upper()}")
        logging.info(f"     SHAP Threshold: {shap_importance_threshold:.4f}")
        logging.info(f"     Permutation Threshold: {permutation_importance_threshold:.4f}")
    logging.info(f"   Optuna Tuning: {'เปิดใช้งาน' if enable_optuna_tuning else 'ปิดใช้งาน'}")
    logging.info(f"   Drift Observer Provided: {'Yes' if drift_observer is not None else 'No'}")
    logging.info(f"   Early Stopping Rounds: {early_stopping_rounds}")

    if output_dir is None or not isinstance(output_dir, str):
        logging.critical("(Error) ไม่ได้ระบุ output_dir หรือไม่ใช่ string.")
        return None, []
    if not os.path.isdir(output_dir):
        try:
            logging.info(f"   สร้าง Output Directory: {output_dir}")
            os.makedirs(output_dir)
        except Exception as e:
            logging.critical(f"(Error) ไม่สามารถสร้าง Output Directory '{output_dir}': {e}", exc_info=True)
            return None, []

    global USE_GPU_ACCELERATION, meta_model_type_used, pattern_label_map
    if enable_optuna_tuning and optuna is None:
        logging.warning("(Warning) ต้องการใช้ Optuna แต่ Library ไม่พร้อมใช้งาน. ปิด Optuna Tuning.")
        enable_optuna_tuning = False
    if model_type_to_train == "catboost" and (CatBoostClassifier is None or Pool is None):
        logging.critical("(Error) ต้องการ Train CatBoost แต่ Library ไม่พร้อมใช้งาน. ไม่สามารถดำเนินการต่อได้.")
        return None, []
    if enable_dynamic_feature_selection and feature_selection_method in ['shap', 'both'] and shap is None:
        logging.warning("(Warning) ต้องการใช้ SHAP สำหรับ Feature Selection แต่ Library ไม่พร้อมใช้งาน. ปิด SHAP selection.")
        if feature_selection_method == 'shap': feature_selection_method = 'permutation'
        elif feature_selection_method == 'both': feature_selection_method = 'permutation'
        if feature_selection_method not in ['permutation']:
            enable_dynamic_feature_selection = False
            logging.warning("   ปิด Dynamic Feature Selection เนื่องจากไม่มี Method ที่รองรับ.")

    task_type_setting = 'GPU' if USE_GPU_ACCELERATION else 'CPU'
    logging.info(f"   GPU Acceleration Available: {USE_GPU_ACCELERATION}. Setting CatBoost task_type to: '{task_type_setting}' (for applicable steps).")
    if task_type_setting == 'GPU':
        logging.info(f"   CatBoost GPU RAM Part setting: {catboost_gpu_ram_part:.2f}")
        logging.info(f"   CatBoost Device setting: 0 (assuming single GPU)")

    trade_log_df = None
    m1_df = None
    if trade_log_df_override is not None and isinstance(trade_log_df_override, pd.DataFrame):
        if trade_log_df_override.empty:
            logging.error(f"(Error) Trade Log Override for '{model_purpose.upper()}' is empty. Cannot proceed with training.")
            return None, []
        required_log_cols_override = ["entry_time", "exit_reason"]
        missing_cols_override = [col for col in required_log_cols_override if col not in trade_log_df_override.columns]
        if missing_cols_override:
            logging.error(f"(Error) Trade Log Override is missing required columns: {missing_cols_override}. Cannot proceed with training.")
            return None, []
        logging.info(f"   ใช้ Trade Log ที่ Filter แล้ว (Override) จำนวน {len(trade_log_df_override)} แถว สำหรับ Model Purpose: {model_purpose.upper()}")
        trade_log_df = trade_log_df_override.copy()
    elif trade_log_path and isinstance(trade_log_path, str):
        logging.info(f"   กำลังโหลด Trade Log (Default Path): {trade_log_path}")
        try:
            trade_log_df = safe_load_csv_auto(trade_log_path)
            if trade_log_df is None:
                raise ValueError("safe_load_csv_auto returned None for trade log.")
            if trade_log_df.empty:
                logging.error(f"(Error) Trade Log (Default Path) for '{model_purpose.upper()}' is empty. Cannot proceed with training.")
                return None, []
            required_log_cols_path = ["entry_time", "exit_reason"]
            missing_cols_path = [col for col in required_log_cols_path if col not in trade_log_df.columns]
            if missing_cols_path:
                logging.error(f"(Error) Trade Log (Default Path) is missing required columns: {missing_cols_path}. Cannot proceed with training.")
                return None, []
            logging.info(f"   โหลด Trade Log (Default) สำเร็จ ({len(trade_log_df)} แถว).")
        except Exception as e:
            logging.error(f"(Error) ไม่สามารถโหลด Trade Log (Default): {e}", exc_info=True)
            return None, []
    else:
        logging.error("(Error) ไม่ได้รับ Trade Log Override และไม่พบไฟล์ Trade Log Path หรือ Path ไม่ถูกต้อง.")
        return None, []

    logging.info("   กำลังประมวลผล Trade Log สำหรับ Training...")
    try:
        time_cols_log = ["entry_time", "close_time", "BE_Triggered_Time"]
        for col in time_cols_log:
            if col in trade_log_df.columns:
                trade_log_df[col] = pd.to_datetime(trade_log_df[col], errors='coerce')
        if "entry_time" not in trade_log_df.columns or not pd.api.types.is_datetime64_any_dtype(trade_log_df["entry_time"]):
            logging.error("(Error) คอลัมน์ 'entry_time' ไม่ใช่ Datetime Type หรือไม่มีใน Trade Log.")
            return None, []
        rows_before_drop = len(trade_log_df)
        trade_log_df.dropna(subset=["entry_time"], inplace=True)
        if len(trade_log_df) < rows_before_drop:
            logging.warning(f"   ลบ {rows_before_drop - len(trade_log_df)} trades ที่มี entry_time ไม่ถูกต้อง.")

        trade_log_df["is_tp"] = (trade_log_df["exit_reason"].astype(str).str.upper() == "TP").astype(int)
        target_dist = trade_log_df['is_tp'].value_counts(normalize=True).round(3)
        logging.info(f"   Target (is_tp from Log) Distribution:\n{target_dist.to_string()}")
        if len(target_dist) < 2:
            logging.warning("   (Warning) Target มีเพียง Class เดียว. Model อาจไม่สามารถ Train ได้อย่างมีความหมาย.")

        if trade_log_df.empty:
            logging.error("(Error) ไม่มี Trades ที่ถูกต้องใน Log หลังการประมวลผล.")
            return None, []
        trade_log_df = trade_log_df.sort_values("entry_time")
        logging.info(f"   ประมวลผล Trade Log สำเร็จ ({len(trade_log_df)} trades).")

    except Exception as e:
        logging.error(f"(Error) เกิดข้อผิดพลาดในการประมวลผล Trade Log: {e}", exc_info=True)
        return None, []

    logging.info(f"   กำลังโหลด M1 Data: {m1_data_path}")
    if not os.path.exists(m1_data_path):
        logging.error(f"(Error) ไม่พบ M1 Data file: {m1_data_path}")
        return None, []

    try:
        m1_df = safe_load_csv_auto(m1_data_path)
        if m1_df is None: raise ValueError("safe_load_csv_auto returned None for M1 data.")
        if m1_df.empty:
            logging.error("   (Error) M1 Data file is empty. Cannot proceed with training.")
            return None, []
        required_m1_features = ["Open", "High", "Low", "Close", "ATR_14"]
        missing_m1_feats = [f for f in required_m1_features if f not in m1_df.columns]
        if missing_m1_feats:
            logging.error(f"(Error) M1 Data is missing required features: {missing_m1_feats}. Cannot proceed with training.")
            return None, []

        logging.info("   กำลังเตรียม Index ของ M1 Data...")
        m1_df.index = pd.to_datetime(m1_df.index, errors='coerce')
        rows_before_drop = len(m1_df)
        m1_df = m1_df[m1_df.index.notna()]
        if len(m1_df) < rows_before_drop:
            logging.warning(f"   ลบ {rows_before_drop - len(m1_df)} แถวที่มี Index เป็น NaT ใน M1 Data.")

        if not isinstance(m1_df.index, pd.DatetimeIndex):
            logging.error("   (Error) ไม่สามารถแปลง M1 index เป็น DatetimeIndex.")
            return None, []
        if m1_df.empty:
            logging.error("   (Error) M1 DataFrame ว่างเปล่าหลังแปลง/ล้าง Index.")
            return None, []
        if not m1_df.index.is_monotonic_increasing:
            logging.info("      Sorting M1 DataFrame index...")
            m1_df = m1_df.sort_index()
        if m1_df.index.has_duplicates:
            dup_count = m1_df.index.duplicated().sum()
            logging.warning(f"   (Warning) พบ Index ซ้ำ {dup_count} รายการใน M1 Data. กำลังลบรายการซ้ำ (เก็บรายการแรก)...")
            m1_df = m1_df[~m1_df.index.duplicated(keep='first')]

        logging.info(f"   โหลดและเตรียม M1 สำเร็จ ({len(m1_df)} แถว). จำนวน Features เริ่มต้น: {len(m1_df.columns)}")
    except Exception as e:
        logging.error(f"(Error) ไม่สามารถโหลดหรือเตรียม M1 data: {e}", exc_info=True)
        return None, []

    logging.info(f"   กำลังเตรียมข้อมูลสำหรับ Meta Model Training (Purpose: {model_purpose.upper()})...")
    merged_df = None
    initial_features_for_selection = [] # Initialize here

    logging.info("   กำลังรวม Trade Log กับ M1 Features (merge_asof)...")
    try:
        if not pd.api.types.is_datetime64_any_dtype(trade_log_df["entry_time"]):
            logging.warning("   Converting trade_log entry_time to datetime again before merge.")
            trade_log_df["entry_time"] = pd.to_datetime(trade_log_df["entry_time"], errors='coerce')
            trade_log_df.dropna(subset=["entry_time"], inplace=True)
        if trade_log_df.empty:
            logging.error("(Error) ไม่มี Trades ที่มี entry_time ถูกต้องหลังการแปลง (ก่อน Merge).")
            return None, []
        if not trade_log_df["entry_time"].is_monotonic_increasing:
            trade_log_df = trade_log_df.sort_values("entry_time")
        if not isinstance(m1_df.index, pd.DatetimeIndex):
            logging.error("   (Error) M1 index is not DatetimeIndex before merge.")
            return None, []
        if not m1_df.index.is_monotonic_increasing:
            logging.warning("   M1 index was not monotonic, sorting again before merge.")
            m1_df = m1_df.sort_index()

        merged_df = pd.merge_asof(
            trade_log_df,
            m1_df,
            left_on="entry_time",
            right_index=True,
            direction="backward",
            tolerance=pd.Timedelta(minutes=5)
        )
        logging.info(f"   Merge completed. Shape after merge: {merged_df.shape}")
        del trade_log_df, m1_df
        gc.collect()

        # Define initial features based on global config or loaded list
        initial_features_for_selection = [f for f in META_CLASSIFIER_FEATURES if f in merged_df.columns]
        if not initial_features_for_selection:
            logging.error("(Error) ไม่มี Features เริ่มต้นที่ใช้ได้ในข้อมูลที่รวมแล้ว.")
            return None, []
        logging.info(f"   Features เริ่มต้นสำหรับการเลือก ({len(initial_features_for_selection)}): {sorted(initial_features_for_selection)}")

        features_to_check_for_nan = initial_features_for_selection + ["is_tp"]
        missing_features_before_dropna = [f for f in features_to_check_for_nan if f not in merged_df.columns]
        if missing_features_before_dropna:
            logging.error(f"(Error) Critical: merged_df ขาด Features ก่อน dropna: {missing_features_before_dropna}")
            return None, []

        rows_before_drop = len(merged_df)
        logging.info(f"   [NaN Check] ก่อน Drop NaN ใน Merged Data (Features/Target): {rows_before_drop} แถว")
        merged_df.dropna(subset=features_to_check_for_nan, inplace=True)
        rows_dropped = rows_before_drop - len(merged_df)
        if rows_dropped > 0:
            logging.info(f"   [NaN Check] ลบ {rows_dropped} Trades ที่มี Missing Features หรือ NaN ใน Features/Target.")
        if merged_df.empty:
            logging.error("(Error) ไม่มีข้อมูลสมบูรณ์หลังการรวมและ Drop NaN.")
            return None, []
        logging.info(f"   (Success) การรวมข้อมูลเสร็จสมบูรณ์ ({len(merged_df)} samples before sampling).")

        if sample_size is not None and sample_size > 0 and sample_size < len(merged_df):
            logging.info(f"   Sampling {sample_size} rows from merged data...")
            merged_df = merged_df.sample(n=sample_size, random_state=42)
            logging.info(f"   (Success) Sampled data size: {len(merged_df)} rows.")
        elif sample_size is not None and sample_size > 0:
            logging.info(f"   (Info) Sample size ({sample_size}) >= data size ({len(merged_df)}). Using all data.")
        elif sample_size is not None and sample_size <= 0:
            logging.warning(f"   (Warning) Invalid sample_size ({sample_size}). Using all data.")

    except Exception as e:
        logging.error(f"(Error) เกิดข้อผิดพลาดระหว่างการรวมข้อมูล: {e}", exc_info=True)
        if 'trade_log_df' in locals() and 'trade_log_df' in globals() and trade_log_df is not None: del trade_log_df
        if 'm1_df' in locals() and 'm1_df' in globals() and m1_df is not None: del m1_df
        if 'merged_df' in locals() and 'merged_df' in globals() and merged_df is not None: del merged_df
        gc.collect()
        return None, []

    selected_features = initial_features_for_selection
    prelim_model = None

    if enable_dynamic_feature_selection and model_type_to_train == "catboost":
        logging.info("\n   --- [Phase 2/B] กำลังดำเนินการ Dynamic Feature Selection ---")
        X_select = merged_df[initial_features_for_selection].copy()
        y_select = merged_df["is_tp"]

        cat_feature_name = 'Pattern_Label'
        categorical_features_select = []
        cat_features_indices_select_cpu = []
        if cat_feature_name in X_select.columns:
            logging.info(f"      จัดการ Categorical Feature: '{cat_feature_name}'...")
            X_select[cat_feature_name] = X_select[cat_feature_name].astype(str).fillna("Normal")
            categorical_features_select = [cat_feature_name]
            try:
                cat_features_indices_select_cpu = [X_select.columns.get_loc(col) for col in categorical_features_select]
                logging.debug(f"         Indices for CatBoost (CPU): {cat_features_indices_select_cpu}")
            except KeyError:
                logging.error(f"      (Error) ไม่พบคอลัมน์ '{cat_feature_name}' ใน X_select columns after processing.")
                categorical_features_select = []
                cat_features_indices_select_cpu = []
        else:
            logging.info(f"      (Info) ไม่พบ Categorical Feature '{cat_feature_name}' สำหรับ Prelim Model.")

        logging.info("      [NaN Check] ตรวจสอบ NaN/Inf ในข้อมูล Feature Selection (X_select)...")
        numeric_cols_select = X_select.select_dtypes(include=np.number).columns
        inf_mask_select = X_select[numeric_cols_select].isin([np.inf, -np.inf])
        if inf_mask_select.any().any():
            cols_inf = numeric_cols_select[inf_mask_select.any()].tolist()
            logging.warning(f"         (Warning) พบ Inf ใน X_select: {cols_inf}. กำลังแทนที่ด้วย NaN...")
            X_select[cols_inf] = X_select[cols_inf].replace([np.inf, -np.inf], np.nan)
        nan_mask_select = X_select[numeric_cols_select].isnull()
        if nan_mask_select.any().any():
            cols_nan = numeric_cols_select[nan_mask_select.any()].tolist()
            logging.warning(f"         (Warning) พบ NaN ใน X_select (Numeric): {cols_nan}. กำลังเติมด้วย ffill().fillna(0)...")
            X_select[cols_nan] = X_select[cols_nan].ffill().fillna(0)

        if X_select.isnull().any().any():
            missing_final = X_select.columns[X_select.isnull().any()].tolist()
            logging.error(f"      (Error) ยังพบ NaN ใน X_select หลังการเติม: {missing_final}")
            enable_dynamic_feature_selection = False
            logging.warning("         ปิด Dynamic Feature Selection เนื่องจากพบ NaN ที่ไม่คาดคิด.")
        else:
            logging.info("         เติม NaN/Inf ใน X_select สำเร็จ.")

        if enable_dynamic_feature_selection:
            logging.info("      กำลัง Train Preliminary Model (สำหรับ Feature Importance)...")
            if prelim_model_params is None:
                prelim_model_params = {
                    'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_seed': 42, 'verbose': 0,
                    'iterations': 500, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3,
                    'early_stopping_rounds': 50, 'auto_class_weights': 'Balanced',
                }
                logging.info("         (ใช้ Default Prelim Params)")
            else:
                logging.info("         (ใช้ Prelim Params ที่กำหนด)")
                prelim_model_params['verbose'] = prelim_model_params.get('verbose', 0)
                prelim_model_params['early_stopping_rounds'] = prelim_model_params.get('early_stopping_rounds', 50)

            prelim_task_type = 'GPU' if USE_GPU_ACCELERATION else 'CPU'
            prelim_model_params['task_type'] = prelim_task_type
            if prelim_task_type == 'GPU':
                prelim_model_params['gpu_ram_part'] = catboost_gpu_ram_part
                prelim_model_params['devices'] = '0'
                logging.info(f"         (Prelim Task Type: {prelim_task_type}, GPU RAM Part: {catboost_gpu_ram_part:.2f}, Device: 0)")
            else:
                logging.info(f"         (Prelim Task Type: {prelim_task_type})")
                prelim_model_params.pop('gpu_ram_part', None)
                prelim_model_params.pop('devices', None)

            prelim_model = CatBoostClassifier(**prelim_model_params)
            try:
                print_gpu_utilization("Before Prelim Fit")
                prelim_model.fit(X_select, y_select, cat_features=cat_features_indices_select_cpu)
                print_gpu_utilization("After Prelim Fit")
                logging.info("      (Success) Train Preliminary Model สำเร็จ.")
            except Exception as e_prelim:
                logging.error(f"      (Error) ไม่สามารถ Train Preliminary Model: {e_prelim}. ข้าม Feature Selection.", exc_info=True)
                enable_dynamic_feature_selection = False

        if enable_dynamic_feature_selection and prelim_model:
            selected_features_shap = set(initial_features_for_selection)
            selected_features_perm = set(initial_features_for_selection)

            if feature_selection_method in ['shap', 'both'] and shap:
                logging.info("      กำลังคำนวณ SHAP Importance...")
                try:
                    explainer_shap = shap.TreeExplainer(prelim_model)
                    shap_pool_select = Pool(X_select, label=y_select, cat_features=cat_features_indices_select_cpu)
                    shap_values_select = explainer_shap.shap_values(shap_pool_select)
                    shap_values_pos_class = None
                    if isinstance(shap_values_select, list) and len(shap_values_select) == 2:
                        shap_values_pos_class = shap_values_select[1]
                    elif isinstance(shap_values_select, np.ndarray) and shap_values_select.ndim == 2:
                        shap_values_pos_class = shap_values_select
                    elif isinstance(shap_values_select, np.ndarray) and shap_values_select.ndim == 3 and shap_values_select.shape[0] >= 2:
                        shap_values_pos_class = shap_values_select[1, :, :]
                    else:
                        logging.warning("      (Warning) ไม่สามารถระบุ SHAP values สำหรับ Positive Class (Prelim).")

                    if shap_values_pos_class is not None:
                        selected_features_shap_list = select_top_shap_features(
                            shap_values_pos_class, initial_features_for_selection,
                            shap_threshold=shap_importance_threshold
                        )
                        selected_features_shap = set(selected_features_shap_list)
                        logging.info(f"      Features ที่เลือกโดย SHAP ({len(selected_features_shap)}): {sorted(list(selected_features_shap))}")
                    else:
                        selected_features_shap = set(initial_features_for_selection)
                        logging.warning("      SHAP calculation for positive class failed. Using all initial features for SHAP step.")
                except Exception as e_shap_select:
                    logging.error(f"      (Error) เกิดข้อผิดพลาดระหว่างคำนวณ SHAP Importance: {e_shap_select}. ใช้ Features ทั้งหมดสำหรับ SHAP.", exc_info=True)
                    selected_features_shap = set(initial_features_for_selection)

            if feature_selection_method in ['permutation', 'both']:
                logging.info("      กำลังคำนวณ Permutation Importance (ใช้ Feature Importance ของ CatBoost)...")
                try:
                    pool_for_perm = shap_pool_select if 'shap_pool_select' in locals() and isinstance(shap_pool_select, Pool) else Pool(X_select, label=y_select, cat_features=cat_features_indices_select_cpu)
                    perm_imp = prelim_model.get_feature_importance(pool_for_perm, type='FeatureImportance')
                    perm_df = pd.DataFrame({'feature': prelim_model.feature_names_, 'perm_importance': perm_imp}).sort_values(by='perm_importance', ascending=False)
                    total_perm_imp = perm_df['perm_importance'].sum()
                    if total_perm_imp > 1e-9:
                        perm_df['normalized_perm_importance'] = perm_df['perm_importance'] / total_perm_imp
                    else:
                        perm_df['normalized_perm_importance'] = 0.0
                        logging.warning("      (Warning) Total Permutation Importance is near zero, cannot normalize.")
                    selected_features_perm_list = perm_df[perm_df['normalized_perm_importance'] >= permutation_importance_threshold]['feature'].tolist()
                    selected_features_perm = set(selected_features_perm_list)
                    logging.info(f"      Features ที่เลือกโดย Permutation ({len(selected_features_perm)}): {sorted(list(selected_features_perm))}")
                    logging.info("         Permutation Importance (Normalized):")
                    logging.info("\n" + perm_df[['feature', 'normalized_perm_importance']].round(5).to_string(index=False))
                    del perm_df
                except Exception as e_perm_select:
                    logging.error(f"      (Error) เกิดข้อผิดพลาดระหว่างคำนวณ Permutation Importance: {e_perm_select}. ใช้ Features ทั้งหมดสำหรับ Permutation.", exc_info=True)
                    selected_features_perm = set(initial_features_for_selection)

            if feature_selection_method == 'shap':
                selected_features = list(selected_features_shap)
            elif feature_selection_method == 'permutation':
                selected_features = list(selected_features_perm)
            elif feature_selection_method == 'both':
                selected_features = list(selected_features_shap.intersection(selected_features_perm))
                if not selected_features:
                    logging.warning("      (Warning) ไม่มี Features ร่วมกันระหว่าง SHAP และ Permutation. กลับไปใช้ SHAP อย่างเดียว.")
                    selected_features = list(selected_features_shap)
                    if not selected_features:
                        logging.error("      (Error) SHAP selection ก็ล้มเหลว. กลับไปใช้ Features เริ่มต้น.")
                        selected_features = initial_features_for_selection
            else:
                selected_features = initial_features_for_selection
                logging.warning(f"      Unknown feature selection method '{feature_selection_method}' or selection failed. Using initial features.")

            logging.info("      พิจารณาเพิ่ม Lag Features ที่สำคัญ...")
            lag_config = {'features': ['Gain_Z', 'Candle_Speed'], 'lags': [1, 3, 5]}
            potential_lag_features = []
            if lag_config:
                for feat_base in lag_config.get('features', []):
                    for lag in lag_config.get('lags', []):
                        lag_feat_name = f"{feat_base}_lag{lag}"
                        if lag_feat_name in merged_df.columns:
                            potential_lag_features.append(lag_feat_name)

            if potential_lag_features:
                logging.info(f"         Lag Features ที่มีให้พิจารณา: {potential_lag_features}")
                try:
                    prelim_fi = prelim_model.get_feature_importance(as_dict=True)
                    significant_lags = []
                    total_fi = sum(prelim_fi.values())
                    fi_threshold_abs = 0.1
                    if total_fi > 1e-9:
                        fi_threshold_norm = 0.001
                        for lag_feat in potential_lag_features:
                            if lag_feat in prelim_fi and (prelim_fi[lag_feat] / total_fi) > fi_threshold_norm:
                                significant_lags.append(lag_feat)
                    else:
                        for lag_feat in potential_lag_features:
                            if lag_feat in prelim_fi and prelim_fi[lag_feat] > fi_threshold_abs:
                                significant_lags.append(lag_feat)

                    if significant_lags:
                        logging.info(f"         Lag Features ที่มีความสำคัญเบื้องต้น: {significant_lags}")
                        added_lags = [lag for lag in significant_lags if lag not in selected_features]
                        if added_lags:
                            logging.info(f"         (Info) เพิ่ม Lag Features ที่มีความสำคัญ: {added_lags}")
                            selected_features.extend(added_lags)
                    else:
                        logging.info("         ไม่มี Lag Features ที่มีความสำคัญเบื้องต้นตามเกณฑ์.")
                    del prelim_fi, significant_lags
                except Exception as e_lag_fi:
                    logging.warning(f"         (Warning) ไม่สามารถประเมินความสำคัญ Lag Features: {e_lag_fi}")
            else:
                logging.info("         ไม่มี Lag Features ให้พิจารณา.")

            selected_features = sorted(list(set(selected_features)))

            if not selected_features:
                logging.error("      (Error) Dynamic Feature Selection ไม่เหลือ Features เลย! กลับไปใช้ Features เริ่มต้น.")
                selected_features = initial_features_for_selection
            else:
                logging.info(f"   --- [Phase 2/B] Final Selected Features ({len(selected_features)}): {sorted(selected_features)} ---")

        logging.debug("      Cleaning up memory after feature selection...")
        del X_select, y_select
        if prelim_model: del prelim_model
        if 'explainer_shap' in locals(): del explainer_shap
        if 'shap_values_select' in locals(): del shap_values_select
        if 'shap_pool_select' in locals(): del shap_pool_select
        if 'pool_for_perm' in locals(): del pool_for_perm
        if 'shap_values_pos_class' in locals(): del shap_values_pos_class
        if 'selected_features_shap' in locals(): del selected_features_shap
        if 'selected_features_perm' in locals(): del selected_features_perm
        gc.collect()
        logging.debug("      Memory cleanup complete.")

    else:
        logging.info("   (Info) ข้าม Dynamic Feature Selection (ปิดใช้งาน หรือ Model ไม่ใช่ CatBoost). ใช้ Features เริ่มต้นทั้งหมด.")
        selected_features = initial_features_for_selection

    logging.info(f"\n   กำลังเตรียมข้อมูล Training สุดท้ายด้วย Features ที่เลือก ({len(selected_features)} ตัว)...")
    if not selected_features:
        logging.error("(Error) ไม่มี Features ที่ถูกเลือกสำหรับ Training.")
        return None, []
    missing_final_features = [f for f in selected_features if f not in merged_df.columns]
    if missing_final_features:
        logging.error(f"(Error) Features ที่เลือก ({missing_final_features}) ไม่พบใน merged_df.")
        return None, []

    X = merged_df[selected_features].copy()
    y = merged_df["is_tp"]

    logging.info("      [NaN Check] ตรวจสอบ NaN/Inf ในข้อมูล Final Training (X)...")
    numeric_cols_X = X.select_dtypes(include=np.number).columns
    inf_mask_X = X[numeric_cols_X].isin([np.inf, -np.inf])
    if inf_mask_X.any().any():
        cols_inf_X = numeric_cols_X[inf_mask_X.any()].tolist()
        logging.warning(f"         (Warning) พบ Inf ใน X (Final): {cols_inf_X}. กำลังแทนที่ด้วย NaN...")
        X[cols_inf_X] = X[cols_inf_X].replace([np.inf, -np.inf], np.nan)
    nan_mask_X = X[numeric_cols_X].isnull()
    if nan_mask_X.any().any():
        cols_nan_X = numeric_cols_X[nan_mask_X.any()].tolist()
        logging.warning(f"         (Warning) พบ NaN ใน X (Numeric - Final): {cols_nan_X}. กำลังเติมด้วย ffill().fillna(0)...")
        X[cols_nan_X] = X[cols_nan_X].ffill().fillna(0)
    if "Pattern_Label" in X.columns:
        X["Pattern_Label"] = X["Pattern_Label"].astype(str)
        if X["Pattern_Label"].isnull().any():
            logging.warning("         (Warning) พบ NaN ใน X ('Pattern_Label' - Final). กำลังเติมด้วย 'Normal'...")
            X["Pattern_Label"] = X["Pattern_Label"].fillna("Normal")
    if "session" in X.columns:
        X["session"] = X["session"].astype(str)
        if X["session"].isnull().any():
            logging.warning("         (Warning) พบ NaN ใน X ('session' - Final). กำลังเติมด้วย 'Other'...")
            X["session"] = X["session"].fillna("Other")

    if X.isnull().any().any():
        missing_final_X = X.columns[X.isnull().any()].tolist()
        logging.error(f"      (Error) ยังพบ NaN ใน X หลังการเติม (Final): {missing_final_X}")
        return None, []
    else:
        logging.info("         เติม NaN/Inf ใน X (Final) สำเร็จ.")

    final_features_before_drop = selected_features[:]
    if features_to_drop_before_train:
        features_actually_dropped = [f for f in features_to_drop_before_train if f in X.columns]
        if features_actually_dropped:
            logging.info(f"      Dropping specified features before final training: {features_actually_dropped}")
            X.drop(columns=features_actually_dropped, inplace=True)
            selected_features = [f for f in selected_features if f not in features_actually_dropped]
            logging.info(f"      Features after drop ({len(selected_features)}): {sorted(selected_features)}")
        else:
            logging.info(f"      (Info) No specified features to drop found in the current feature set.")

    try:
        logging.info("      [RAM Opt] Converting Final Training Features (X) to float32 (Before Split)...")
        numeric_cols_train_pre_split = X.select_dtypes(include=np.number).columns
        converted_final = 0
        for col in numeric_cols_train_pre_split:
            if col == 'cluster': continue
            if pd.api.types.is_float_dtype(X[col].dtype):
                try:
                    X[col] = pd.to_numeric(X[col], downcast='float')
                    if X[col].dtype == 'float32':
                        converted_final += 1
                except Exception as e_astype_train:
                    logging.warning(f"         (Warning) ไม่สามารถแปลงคอลัมน์ '{col}' เป็น float32: {e_astype_train}. ใช้ float64 ต่อไป.")
        logging.info(f"      (Success) แปลง {converted_final} Final Training Features เป็น float32 (เท่าที่ทำได้) สำเร็จ.")
    except Exception as e_astype_train_block:
        logging.warning(f"      (Warning) ไม่สามารถแปลง Final Training Features เป็น float32: {e_astype_train_block}")

    best_params_from_optuna = None
    if enable_optuna_tuning and optuna is not None and model_type_to_train == "catboost":
        logging.info("\n   --- [Phase 3/A] กำลังดำเนินการ Hyperparameter Optimization (Optuna) ---")
        logging.warning("      (Info) Optuna logic is defined but currently disabled by ENABLE_OPTUNA_TUNING=False.")
    else:
        logging.info("\n   --- [Phase 3/A] ข้าม Hyperparameter Optimization (Optuna ปิดใช้งาน หรือ Model ไม่ใช่ CatBoost) ---")

    trained_models = {}
    shap_values_cat_val = None
    X_val_cat_for_shap = None
    final_features_catboost = selected_features
    cat_model = None # Initialize cat_model to None

    if model_type_to_train == "catboost" and CatBoostClassifier and Pool:
        logging.info(f"\n   --- Training Final CatBoost Model (Purpose: {model_purpose.upper()}) ---")
        X_train_cat, X_val_cat, y_train_cat, y_val_cat = None, None, None, None # Initialize split variables
        try:
            X_cat = X.copy()
            cat_feature_name_final = 'Pattern_Label'
            categorical_features_cat_final = []
            cat_features_indices_cpu_final = []
            if cat_feature_name_final in X_cat.columns:
                logging.info(f"      จัดการ Categorical Feature (Final): '{cat_feature_name_final}'...")
                X_cat[cat_feature_name_final] = X_cat[cat_feature_name_final].astype(str).fillna("Normal")
                categorical_features_cat_final = [cat_feature_name_final]
                try:
                    cat_features_indices_cpu_final = [X_cat.columns.get_loc(col) for col in categorical_features_cat_final]
                    logging.debug(f"         Indices for CatBoost (CPU - Final): {cat_features_indices_cpu_final}")
                except KeyError:
                    logging.error(f"      (Error) ไม่พบคอลัมน์ '{cat_feature_name_final}' ใน X_cat columns (Final).")
                    categorical_features_cat_final = []
                    cat_features_indices_cpu_final = []
            else:
                logging.info("      (Info) ไม่พบ 'Pattern_Label' ใน Features ที่เลือกสุดท้าย.")

            logging.info("      Splitting data into Train/Validation sets (80/20 stratified)...")
            # <<< [Patch] MODIFIED v4.8.8 (Patch 2): Added check for sufficient samples >>>
            if len(X_cat) < 5: # Need at least a few samples for split
                logging.error(f"      (Error) Not enough samples ({len(X_cat)}) to perform train/validation split.")
                raise ValueError("Insufficient samples for train/val split")
            if y.nunique() < 2:
                logging.warning("      (Warning) Only one class present in target 'y'. Stratified split might fail or be meaningless. Using non-stratified split.")
                X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(
                    X_cat, y, test_size=0.2, random_state=42
                )
            else:
                try:
                    X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(
                        X_cat, y, test_size=0.2, random_state=42, stratify=y
                    )
                except ValueError as e_split: # Catch potential errors if stratification fails
                    logging.warning(f"      (Warning) Stratified split failed ({e_split}). Falling back to non-stratified split.")
                    X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(
                        X_cat, y, test_size=0.2, random_state=42
                    )
            # <<< End of [Patch] MODIFIED v4.8.8 (Patch 2) >>>

            logging.info(f"      Train Size (Final): {len(X_train_cat)}, Validation Size (Final): {len(X_val_cat)}")
            val_target_dist = y_val_cat.value_counts(normalize=True).round(3)
            logging.info(f"      Validation Target Distribution (Final):\n{val_target_dist.to_string()}")
            if len(val_target_dist) < 2:
                logging.warning("      (Warning) Validation set has only one class. Evaluation metrics like AUC might be undefined.")

            logging.info("      [RAM Opt] Converting Train/Validation sets to float32 (After Split)...")
            try:
                for col in X_train_cat.select_dtypes(include='float64').columns:
                    X_train_cat[col] = X_train_cat[col].astype('float32')
                logging.debug("         Train set converted to float32.")
            except Exception as e_astype_train:
                logging.warning(f"         (Warning) Could not convert Train set to float32: {e_astype_train}")
            try:
                for col in X_val_cat.select_dtypes(include='float64').columns:
                    X_val_cat[col] = X_val_cat[col].astype('float32')
                logging.debug("         Validation set converted to float32.")
            except Exception as e_astype_val:
                logging.warning(f"         (Warning) Could not convert Validation set to float32: {e_astype_val}")

            X_val_cat_for_shap = X_val_cat.copy() # Keep copy before potential Pool conversion

            logging.info("      กำลังกำหนด Final Model Parameters...")
            if best_params_from_optuna:
                logging.info("         Using Best Params from Optuna.")
                final_model_params_to_use = best_params_from_optuna
                final_model_params_to_use.setdefault('loss_function', 'Logloss')
                final_model_params_to_use.setdefault('eval_metric', 'AUC')
                final_model_params_to_use.setdefault('random_seed', 42)
                final_model_params_to_use.setdefault('early_stopping_rounds', early_stopping_rounds)
                final_model_params_to_use.setdefault('verbose', 100)
                final_model_params_to_use.setdefault('auto_class_weights', 'Balanced')
            else:
                logging.info("         Using Fixed Params (v4.7.1)...")
                final_model_params_to_use = {
                    'iterations': 3000, 'learning_rate': 0.01, 'depth': 4, 'l2_leaf_reg': 30,
                    'eval_metric': "AUC", 'auto_class_weights': "Balanced", 'early_stopping_rounds': early_stopping_rounds,
                    'random_seed': 42, 'verbose': 100, 'loss_function': 'Logloss',
                }

            final_model_params_to_use['task_type'] = task_type_setting
            if task_type_setting == 'GPU':
                final_model_params_to_use['gpu_ram_part'] = catboost_gpu_ram_part
                final_model_params_to_use['devices'] = '0'
            else:
                final_model_params_to_use.pop('gpu_ram_part', None)
                final_model_params_to_use.pop('devices', None)

            logging.info(f"         Final Params: {final_model_params_to_use}")

            cat_model = CatBoostClassifier(**final_model_params_to_use)
            logging.info(f"      Fitting Final CatBoost model (Purpose: {model_purpose.upper()})...");
            print_gpu_utilization("Before Final Fit")
            fit_start_time = time.time()

            # <<< [Patch] MODIFIED v4.8.8 (Patch 2): Create Pool for eval_set >>>
            eval_pool = None
            if X_val_cat is not None and y_val_cat is not None and not X_val_cat.empty:
                try:
                    eval_pool = Pool(X_val_cat, label=y_val_cat, cat_features=cat_features_indices_cpu_final)
                    logging.debug("         Created Pool for eval_set.")
                except Exception as e_pool_eval:
                    logging.error(f"         (Error) Failed to create Pool for eval_set: {e_pool_eval}. Fitting without eval_set.")
                    eval_pool = None # Fallback to fitting without eval_set
            # <<< End of [Patch] MODIFIED v4.8.8 (Patch 2) >>>

            # <<< [Patch] MODIFIED v4.8.8 (Patch 2): Use Pool in fit if created >>>
            cat_model.fit(
                X_train_cat, y_train_cat,
                cat_features=cat_features_indices_cpu_final,
                eval_set=eval_pool # Use the created Pool object here
            )
            # <<< End of [Patch] MODIFIED v4.8.8 (Patch 2) >>>

            fit_duration = time.time() - fit_start_time
            print_gpu_utilization("After Final Fit")
            logging.info(f"      (Success) Final CatBoost training (Purpose: {model_purpose.upper()}) finished in {fit_duration:.2f} seconds.")
            trained_models["catboost"] = cat_model
            meta_model_type_used = cat_model.__class__.__name__

            # --- Evaluate Final Model on Validation Set ---
            # <<< [Patch] MODIFIED v4.8.8 (Patch 2): Moved evaluation inside try block >>>
            logging.info("\n      --- Model Quality Check (Final CatBoost Model) ---")
            try:
                # Check overfitting using only validation data (train data deleted)
                # Pass y_val_cat which is guaranteed to exist if fit succeeded
                check_model_overfit(cat_model, None, None, X_val_cat_for_shap, y_val_cat, metric="AUC", threshold_pct=15.0)
                check_model_overfit(cat_model, None, None, X_val_cat_for_shap, y_val_cat, metric="LogLoss", threshold_pct=15.0)

                logging.info(f"      Validation Metrics (Final Model - Purpose: {model_purpose.upper()}):");
                y_pred_cat_val = cat_model.predict(X_val_cat_for_shap)
                y_proba_cat_val_raw = cat_model.predict_proba(X_val_cat_for_shap)
                y_proba_cat_val = y_proba_cat_val_raw[:, 1] if y_proba_cat_val_raw is not None and y_proba_cat_val_raw.ndim == 2 and y_proba_cat_val_raw.shape[1] >= 2 else None

                val_accuracy = accuracy_score(y_val_cat, y_pred_cat_val) if y_pred_cat_val is not None else np.nan
                logging.info(f"         Accuracy:  {val_accuracy:.4f}");
                val_auc = np.nan
                if y_proba_cat_val is not None:
                    try:
                        val_auc = roc_auc_score(y_val_cat, y_proba_cat_val)
                        logging.info(f"         AUC:       {val_auc:.4f}");
                    except ValueError as e_auc_val:
                        logging.warning(f"         AUC:       Error ({e_auc_val})")
                else:
                    logging.warning("         AUC:       Cannot calculate (Invalid probabilities)")

                val_logloss = np.nan
                if y_proba_cat_val_raw is not None:
                    try:
                        val_logloss = log_loss(y_val_cat, y_proba_cat_val_raw, labels=cat_model.classes_)
                        logging.info(f"         LogLoss:   {val_logloss:.4f}")
                    except ValueError as e_ll_val:
                        logging.warning(f"         LogLoss:   Error ({e_ll_val})")
                else:
                    logging.warning("         LogLoss:   Cannot calculate (Invalid probabilities)")

                # Classification Report
                if y_pred_cat_val is not None:
                    report = classification_report(y_val_cat, y_pred_cat_val, target_names=['Not TP', 'TP'], zero_division=0)
                    logging.info(f"         Classification Report:\n{report}")
                else:
                    logging.warning("         Classification Report: Cannot generate (Invalid predictions)")

                # Performance Alert
                if pd.notna(val_auc) and val_auc < 0.65:
                    logging.critical(f"      (CRITICAL WARNING) Final Model Validation AUC ({val_auc:.4f}) is below target (0.65)!")
                elif pd.notna(val_auc) and val_auc < 0.70:
                    logging.warning(f"      (Warning) Final Model Validation AUC ({val_auc:.4f}) is below desired target (0.70).")

            except Exception as e_quality_check:
                logging.error(f"      (Error) Error during Final Model Quality Check: {e_quality_check}", exc_info=True)

            # --- SHAP Analysis on Validation Set ---
            if shap and X_val_cat_for_shap is not None and not X_val_cat_for_shap.empty and cat_model is not None: # Check cat_model exists
                logging.info(f"\n      --- SHAP Analysis (Final Model - Validation Set - Purpose: {model_purpose.upper()}) ---")
                try:
                    analyze_feature_importance_shap(
                        cat_model, meta_model_type_used, X_val_cat_for_shap,
                        final_features_catboost, output_dir
                    )
                    logging.info("\n         --- Feature Noise Check (SHAP - Final Model - Validation Set) ---")
                    logging.info("            Recalculating SHAP values for Noise Check (Validation Set)...")
                    shap_pool_val_final = Pool(X_val_cat_for_shap, label=y_val_cat, cat_features=cat_features_indices_cpu_final)
                    explainer_val_final = shap.TreeExplainer(cat_model)
                    shap_values_val_final = explainer_val_final.shap_values(shap_pool_val_final)
                    shap_values_cat_val = None

                    if isinstance(shap_values_val_final, list) and len(shap_values_val_final) == 2:
                        shap_values_cat_val = shap_values_val_final[1]
                    elif isinstance(shap_values_val_final, np.ndarray) and shap_values_val_final.ndim == 2:
                        shap_values_cat_val = shap_values_val_final
                    elif isinstance(shap_values_val_final, np.ndarray) and shap_values_val_final.ndim == 3 and shap_values_val_final.shape[0] >= 2:
                        shap_values_cat_val = shap_values_val_final[1, :, :]

                    if shap_values_cat_val is not None:
                        check_feature_noise_shap(shap_values_cat_val, final_features_catboost, threshold=shap_importance_threshold)
                    else:
                        logging.warning("         (Warning) ไม่สามารถระบุ SHAP values สำหรับ TP Class (Final Validation) สำหรับ Noise Check.")
                    del shap_pool_val_final, explainer_val_final, shap_values_val_final, shap_values_cat_val
                    gc.collect()
                except Exception as e_shap:
                    logging.error(f"         (Error) Error during SHAP Analysis/Noise Check (Final Model): {e_shap}", exc_info=True)
            else:
                logging.warning("      (Warning) ข้าม SHAP Analysis (Final Model): Library/Data ไม่พร้อม, Validation Set ว่างเปล่า, หรือ Model Train ไม่สำเร็จ.")
            # <<< End of [Patch] MODIFIED v4.8.8 (Patch 2) >>>

        except Exception as e:
            logging.error(f"      (Error) Error during Final CatBoost training/evaluation: {e}", exc_info=True)
            # Ensure cleanup even if error occurs mid-training
            if 'X_train_cat' in locals(): del X_train_cat
            if 'y_train_cat' in locals(): del y_train_cat
            if 'X_val_cat' in locals(): del X_val_cat
            if 'y_val_cat' in locals(): del y_val_cat # y_val_cat might not be assigned if split fails
            if 'X_cat' in locals(): del X_cat
            if 'X' in locals(): del X
            if 'y' in locals(): del y
            if 'merged_df' in locals() and 'merged_df' in globals() and merged_df is not None: del merged_df
            gc.collect()

    # --- Save Final Model and Features ---
    logging.info(f"\n   --- Saving Final Model (Purpose: {model_purpose.upper()}) ---")
    saved_model_paths = {}
    if not trained_models:
        logging.warning("   (Warning) ไม่มี Models ที่ Train สำเร็จให้ Save.")
        # <<< [Patch] MODIFIED v4.8.8 (Patch 2): Return selected_features even if training fails >>>
        return None, selected_features
        # <<< End of [Patch] MODIFIED v4.8.8 (Patch 2) >>>

    for model_name, model_obj in trained_models.items():
        if model_name == "catboost":
            if model_purpose == 'main': model_filename = META_CLASSIFIER_PATH
            elif model_purpose == 'spike': model_filename = SPIKE_MODEL_PATH
            elif model_purpose == 'cluster': model_filename = CLUSTER_MODEL_PATH
            else: model_filename = f"meta_classifier_{model_purpose}.pkl"

            model_path = os.path.join(output_dir, model_filename)
            try:
                joblib_dump(model_obj, model_path)
                logging.info(f"      (Success) Saved Final {model_name.upper()} (Purpose: {model_purpose.upper()}): {model_path}")
                saved_model_paths[model_purpose] = model_path
            except Exception as e:
                logging.error(f"      (Error) Failed to save Final {model_name.upper()} (Purpose: {model_purpose.upper()}): {e}", exc_info=True)
        else:
            logging.warning(f"   (Warning) ข้ามการ Save สำหรับ Model Type ที่ไม่คาดคิด: {model_name}")

    if model_purpose == 'main':
        default_model_path = os.path.join(output_dir, META_CLASSIFIER_PATH)
        if os.path.exists(default_model_path):
            logging.info(f"   (Info) Main model saved as default ({META_CLASSIFIER_PATH}). No separate linking needed.")
        else:
            logging.warning(f"\n   (Warning) Main model file ({META_CLASSIFIER_PATH}) not found after saving attempt.")

    features_filename = f"features_{model_purpose}.json"
    features_file_path = os.path.join(output_dir, features_filename)
    try:
        logging.info(f"   Saving final selected features ({len(final_features_catboost)}) for '{model_purpose}' to: {features_file_path}")
        with open(features_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_features_catboost, f, indent=4, default=simple_converter)
        logging.info(f"   (Success) Saved final features list for '{model_purpose}'.")
    except Exception as e_save_feat:
        logging.error(f"   (Error) Failed to save final features list for '{model_purpose}': {e_save_feat}", exc_info=True)

    end_train_time = time.time()
    logging.info(f"(Finished - v4.8.8 Patch 2) Meta Classifier Training (Purpose: {model_purpose.upper()}) complete in {end_train_time - start_train_time:.2f} seconds.") # Updated version in log
    if 'X_val_cat_for_shap' in locals(): del X_val_cat_for_shap
    gc.collect()
    return saved_model_paths, final_features_catboost

logging.info("Part 7: Model Training Function Loaded (v4.8.8 Patch 2 Applied).")
# === END OF PART 7/12 ===


# === START OF PART 8/12 ===

# ==============================================================================
# === PART 8: Backtesting Engine (v4.8.8 - Patch 26.6.1 Applied) ===
# ==============================================================================
# <<< MODIFIED v4.8.8 (Patch 20): Implemented Refactoring Plan: Separated Exit Check and State Update functions. >>>
# <<< MODIFIED v4.8.8 (Patch 21): Fixed NameError for counters in _update_open_order_state by passing them as arguments. >>>
# <<< MODIFIED v4.8.8 (Patch 22.1): Fixed Exit Reason & Price Finalization for End-of-Period Close. >>>
# <<< MODIFIED v4.8.8 (Patch 23): Fixed BE/TSL Priority, moved BE-SL Refine logic, removed EoP check logic, adjusted TP Fixture logic in test. >>>
# <<< MODIFIED v4.8.8 (Patch 24): Fixed _check_order_exit_conditions return, SL Hit check, BE-SL PnL Calc; Reverted EoP logic; Adjusted test fixtures. >>>
# <<< MODIFIED v4.8.8 (Patch 25): Added price tolerance check (math.isclose), refined BE-SL PnL calc, verified EoP logic. >>>
# <<< MODIFIED v4.8.8 (Patch 26.2): Applied math.isclose, BE-SL PnL Fix, SL Multiplier check, refined EoP logic, and improved logging. >>>
# <<< MODIFIED v4.8.8 (Patch 26.3.1): Applied [PATCH B] to _check_order_exit_conditions (new name & logic), verified BE-SL PnL and SL multiplier usage. >>>
# <<< MODIFIED v4.8.8 (Patch 26.4.1): Unified [PATCH B] for logging in _update_open_order_state, and [PATCH C] for error handling in run_backtest_simulation_v34. >>>
# <<< MODIFIED v4.8.8 (Patch 26.6.1): Applied user-provided fixes for f-string formatting in logging statements to prevent ValueError. >>>
import logging
import pandas as pd
import numpy as np
import random
import time
from collections import defaultdict
import gc # For memory management
import math # For math.isclose
import importlib # Added for safe global access
import sys # Added for safe global access
import traceback # Added for [PATCH C]

# Ensure tqdm is available (imported in Part 1)
try:
    from tqdm.notebook import tqdm
except ImportError:
    tqdm = None # Define tqdm as None if import fails

# --- Safe Global Variable Access ---
# Use a helper function for safe access with defaults
# Assuming safe_get_global is defined in Part 3
# def safe_get_global(var_name, default_value): ...

# Define defaults (these should match Part 2 ideally)
DEFAULT_SESSION_TIMES_UTC = {"Asia": (0, 8), "London": (7, 16), "NY": (13, 21)}
DEFAULT_BASE_TP_MULTIPLIER = 1.8
DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO = 1.8
DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO = 0.75
DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R = 0.5
DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R = 1.0
DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R = 0.3
DEFAULT_ADAPTIVE_TSL_START_ATR_MULT = 1.5
DEFAULT_ENABLE_SPIKE_GUARD = True
DEFAULT_MIN_SIGNAL_SCORE_ENTRY = 2.0
DEFAULT_RECOVERY_MODE_CONSECUTIVE_LOSSES = 4
DEFAULT_RECOVERY_MODE_LOT_MULTIPLIER = 0.5
DEFAULT_MIN_LOT_SIZE = 0.01
DEFAULT_MAX_LOT_SIZE = 5.0
DEFAULT_POINT_VALUE = 0.1
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_USE_REENTRY = True
DEFAULT_REENTRY_COOLDOWN_BARS = 1
DEFAULT_TIMEFRAME_MINUTES_M1 = 1
DEFAULT_MAX_CONCURRENT_ORDERS = 5
DEFAULT_MAX_HOLDING_BARS = 24
DEFAULT_COMMISSION_PER_001_LOT = 0.10
DEFAULT_SPREAD_POINTS = 2.0
DEFAULT_MIN_SLIPPAGE_POINTS = -5.0
DEFAULT_MAX_SLIPPAGE_POINTS = -1.0
DEFAULT_MAX_DRAWDOWN_THRESHOLD = 0.30
DEFAULT_ENABLE_FORCED_ENTRY = True
DEFAULT_FORCED_ENTRY_BAR_THRESHOLD = 100
DEFAULT_FORCED_ENTRY_MIN_SIGNAL_SCORE = 0.5
DEFAULT_FORCED_ENTRY_LOOKBACK_PERIOD = 500
DEFAULT_FORCED_ENTRY_CHECK_MARKET_COND = True
DEFAULT_FORCED_ENTRY_MAX_ATR_MULT = 2.5
DEFAULT_FORCED_ENTRY_MIN_GAIN_Z_ABS = 1.0
DEFAULT_FORCED_ENTRY_ALLOWED_REGIMES = ["Normal", "Breakout", "StrongTrend"]
DEFAULT_FE_ML_FILTER_THRESHOLD = 0.40
DEFAULT_forced_entry_max_consecutive_losses = 2
DEFAULT_min_equity_threshold_pct = 0.70
DEFAULT_ENTRY_CONFIG_PER_FOLD = {0: {"sl_multiplier": 2.0, "gain_z_thresh": 0.3, "cooldown_sec": 0, "min_signal_score": 2.0}}
DEFAULT_BASE_BE_SL_R_THRESHOLD = 1.0
DEFAULT_DYNAMIC_BE_ATR_THRESHOLD_HIGH = 1.2
DEFAULT_DYNAMIC_BE_R_ADJUST_HIGH = 0.2
DEFAULT_ENABLE_PARTIAL_TP = True
DEFAULT_PARTIAL_TP_LEVELS = [{"r_multiple": 0.8, "close_pct": 0.5}]
DEFAULT_PARTIAL_TP_MOVE_SL_TO_ENTRY = True
DEFAULT_ENABLE_KILL_SWITCH = True
DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD = 0.20
DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 7
DEFAULT_FUND_PROFILES = {"NORMAL": {"risk": 0.01, "mm_mode": "balanced"}}
DEFAULT_FUND_NAME = "NORMAL"
DEFAULT_USE_META_CLASSIFIER = True
DEFAULT_META_MIN_PROBA_THRESH = 0.55
DEFAULT_REENTRY_MIN_PROBA_THRESH = 0.55
DEFAULT_OUTPUT_DIR = "./output_default"

# Access globals safely using safe_get_global (defined in Part 3)
SESSION_TIMES_UTC = safe_get_global('SESSION_TIMES_UTC', DEFAULT_SESSION_TIMES_UTC)
BASE_TP_MULTIPLIER = safe_get_global('BASE_TP_MULTIPLIER', DEFAULT_BASE_TP_MULTIPLIER)
ADAPTIVE_TSL_HIGH_VOL_RATIO = safe_get_global('ADAPTIVE_TSL_HIGH_VOL_RATIO', DEFAULT_ADAPTIVE_TSL_HIGH_VOL_RATIO)
ADAPTIVE_TSL_LOW_VOL_RATIO = safe_get_global('ADAPTIVE_TSL_LOW_VOL_RATIO', DEFAULT_ADAPTIVE_TSL_LOW_VOL_RATIO)
ADAPTIVE_TSL_DEFAULT_STEP_R = safe_get_global('ADAPTIVE_TSL_DEFAULT_STEP_R', DEFAULT_ADAPTIVE_TSL_DEFAULT_STEP_R)
ADAPTIVE_TSL_HIGH_VOL_STEP_R = safe_get_global('ADAPTIVE_TSL_HIGH_VOL_STEP_R', DEFAULT_ADAPTIVE_TSL_HIGH_VOL_STEP_R)
ADAPTIVE_TSL_LOW_VOL_STEP_R = safe_get_global('ADAPTIVE_TSL_LOW_VOL_STEP_R', DEFAULT_ADAPTIVE_TSL_LOW_VOL_STEP_R)
ADAPTIVE_TSL_START_ATR_MULT = safe_get_global('ADAPTIVE_TSL_START_ATR_MULT', DEFAULT_ADAPTIVE_TSL_START_ATR_MULT)
ENABLE_SPIKE_GUARD = safe_get_global('ENABLE_SPIKE_GUARD', DEFAULT_ENABLE_SPIKE_GUARD)
MIN_SIGNAL_SCORE_ENTRY = safe_get_global('MIN_SIGNAL_SCORE_ENTRY', DEFAULT_MIN_SIGNAL_SCORE_ENTRY)
RECOVERY_MODE_CONSECUTIVE_LOSSES = safe_get_global('RECOVERY_MODE_CONSECUTIVE_LOSSES', DEFAULT_RECOVERY_MODE_CONSECUTIVE_LOSSES)
RECOVERY_MODE_LOT_MULTIPLIER = safe_get_global('RECOVERY_MODE_LOT_MULTIPLIER', DEFAULT_RECOVERY_MODE_LOT_MULTIPLIER)
MIN_LOT_SIZE = safe_get_global('MIN_LOT_SIZE', DEFAULT_MIN_LOT_SIZE)
MAX_LOT_SIZE = safe_get_global('MAX_LOT_SIZE', DEFAULT_MAX_LOT_SIZE)
POINT_VALUE = safe_get_global('POINT_VALUE', DEFAULT_POINT_VALUE)
DEFAULT_RISK_PER_TRADE = safe_get_global('DEFAULT_RISK_PER_TRADE', DEFAULT_RISK_PER_TRADE)
USE_REENTRY = safe_get_global('USE_REENTRY', DEFAULT_USE_REENTRY)
REENTRY_COOLDOWN_BARS = safe_get_global('REENTRY_COOLDOWN_BARS', DEFAULT_REENTRY_COOLDOWN_BARS)
TIMEFRAME_MINUTES_M1 = safe_get_global('TIMEFRAME_MINUTES_M1', DEFAULT_TIMEFRAME_MINUTES_M1)
MAX_CONCURRENT_ORDERS = safe_get_global('MAX_CONCURRENT_ORDERS', DEFAULT_MAX_CONCURRENT_ORDERS)
MAX_HOLDING_BARS = safe_get_global('MAX_HOLDING_BARS', DEFAULT_MAX_HOLDING_BARS)
COMMISSION_PER_001_LOT = safe_get_global('COMMISSION_PER_001_LOT', DEFAULT_COMMISSION_PER_001_LOT)
SPREAD_POINTS = safe_get_global('SPREAD_POINTS', DEFAULT_SPREAD_POINTS)
MIN_SLIPPAGE_POINTS = safe_get_global('MIN_SLIPPAGE_POINTS', DEFAULT_MIN_SLIPPAGE_POINTS)
MAX_SLIPPAGE_POINTS = safe_get_global('MAX_SLIPPAGE_POINTS', DEFAULT_MAX_SLIPPAGE_POINTS)
MAX_DRAWDOWN_THRESHOLD = safe_get_global('MAX_DRAWDOWN_THRESHOLD', DEFAULT_MAX_DRAWDOWN_THRESHOLD)
ENABLE_FORCED_ENTRY = safe_get_global('ENABLE_FORCED_ENTRY', DEFAULT_ENABLE_FORCED_ENTRY)
FORCED_ENTRY_BAR_THRESHOLD = safe_get_global('FORCED_ENTRY_BAR_THRESHOLD', DEFAULT_FORCED_ENTRY_BAR_THRESHOLD)
FORCED_ENTRY_MIN_SIGNAL_SCORE = safe_get_global('FORCED_ENTRY_MIN_SIGNAL_SCORE', DEFAULT_FORCED_ENTRY_MIN_SIGNAL_SCORE)
FORCED_ENTRY_LOOKBACK_PERIOD = safe_get_global('FORCED_ENTRY_LOOKBACK_PERIOD', DEFAULT_FORCED_ENTRY_LOOKBACK_PERIOD)
FORCED_ENTRY_CHECK_MARKET_COND = safe_get_global('FORCED_ENTRY_CHECK_MARKET_COND', DEFAULT_FORCED_ENTRY_CHECK_MARKET_COND)
FORCED_ENTRY_MAX_ATR_MULT = safe_get_global('FORCED_ENTRY_MAX_ATR_MULT', DEFAULT_FORCED_ENTRY_MAX_ATR_MULT)
FORCED_ENTRY_MIN_GAIN_Z_ABS = safe_get_global('FORCED_ENTRY_MIN_GAIN_Z_ABS', DEFAULT_FORCED_ENTRY_MIN_GAIN_Z_ABS)
FORCED_ENTRY_ALLOWED_REGIMES = safe_get_global('FORCED_ENTRY_ALLOWED_REGIMES', DEFAULT_FORCED_ENTRY_ALLOWED_REGIMES)
FE_ML_FILTER_THRESHOLD = safe_get_global('FE_ML_FILTER_THRESHOLD', DEFAULT_FE_ML_FILTER_THRESHOLD)
forced_entry_max_consecutive_losses = safe_get_global('forced_entry_max_consecutive_losses', 2)
min_equity_threshold_pct = safe_get_global('min_equity_threshold_pct', DEFAULT_min_equity_threshold_pct)
ENTRY_CONFIG_PER_FOLD = safe_get_global('ENTRY_CONFIG_PER_FOLD', DEFAULT_ENTRY_CONFIG_PER_FOLD)
BASE_BE_SL_R_THRESHOLD = safe_get_global('BASE_BE_SL_R_THRESHOLD', DEFAULT_BASE_BE_SL_R_THRESHOLD)
DYNAMIC_BE_ATR_THRESHOLD_HIGH = safe_get_global('DYNAMIC_BE_ATR_THRESHOLD_HIGH', DEFAULT_DYNAMIC_BE_ATR_THRESHOLD_HIGH)
DYNAMIC_BE_R_ADJUST_HIGH = safe_get_global('DYNAMIC_BE_R_ADJUST_HIGH', DEFAULT_DYNAMIC_BE_R_ADJUST_HIGH)
ENABLE_PARTIAL_TP = safe_get_global('ENABLE_PARTIAL_TP', DEFAULT_ENABLE_PARTIAL_TP)
PARTIAL_TP_LEVELS = safe_get_global('PARTIAL_TP_LEVELS', DEFAULT_PARTIAL_TP_LEVELS)
PARTIAL_TP_MOVE_SL_TO_ENTRY = safe_get_global('PARTIAL_TP_MOVE_SL_TO_ENTRY', DEFAULT_PARTIAL_TP_MOVE_SL_TO_ENTRY)
ENABLE_KILL_SWITCH = safe_get_global('ENABLE_KILL_SWITCH', DEFAULT_ENABLE_KILL_SWITCH)
KILL_SWITCH_MAX_DD_THRESHOLD = safe_get_global('KILL_SWITCH_MAX_DD_THRESHOLD', DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD)
KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = safe_get_global('KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD', DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD)
FUND_PROFILES = safe_get_global('FUND_PROFILES', DEFAULT_FUND_PROFILES)
DEFAULT_FUND_NAME = safe_get_global('DEFAULT_FUND_NAME', DEFAULT_FUND_NAME)
USE_META_CLASSIFIER = safe_get_global('USE_META_CLASSIFIER', DEFAULT_USE_META_CLASSIFIER)
META_MIN_PROBA_THRESH = safe_get_global('META_MIN_PROBA_THRESH', DEFAULT_META_MIN_PROBA_THRESH)
REENTRY_MIN_PROBA_THRESH = safe_get_global('REENTRY_MIN_PROBA_THRESH', DEFAULT_REENTRY_MIN_PROBA_THRESH)
OUTPUT_DIR = safe_get_global('OUTPUT_DIR', DEFAULT_OUTPUT_DIR)


# --- Backtesting Helper Functions ---
# safe_set_datetime is now in Part 3

def get_session_tag(timestamp, session_times_utc=None):
    """Helper to get the trading session tag based on UTC timestamp."""
    if session_times_utc is None:
        global SESSION_TIMES_UTC
        session_times_utc = SESSION_TIMES_UTC
    if pd.isna(timestamp):
        return "N/A"
    try:
        if timestamp.tzinfo is None:
            ts_utc = pd.Timestamp(timestamp, tz='UTC')
        else:
            ts_utc = timestamp.tz_convert('UTC')

        hour = ts_utc.hour
        sessions = []
        for name, (start, end) in session_times_utc.items():
            if start <= end:
                if start <= hour < end:
                    sessions.append(name)
            else:
                if hour >= start or hour < end:
                    sessions.append(name)
        return "/".join(sorted(sessions)) if sessions else "Other"
    except Exception as e:
        logging.error(f"   (Error) Error in get_session_tag for {timestamp}: {e}", exc_info=True)
        return "Error"

def dynamic_tp2_multiplier(current_atr, avg_atr, base=None):
    """Calculates a dynamic TP multiplier based on current vs average ATR."""
    if base is None:
        global BASE_TP_MULTIPLIER
        base = BASE_TP_MULTIPLIER
    current_atr_num = pd.to_numeric(current_atr, errors='coerce')
    avg_atr_num = pd.to_numeric(avg_atr, errors='coerce')
    if pd.isna(current_atr_num) or pd.isna(avg_atr_num) or np.isinf(current_atr_num) or np.isinf(avg_atr_num) or avg_atr_num < 1e-9:
        return base
    try:
        ratio = current_atr_num / avg_atr_num
        high_vol_ratio = ADAPTIVE_TSL_HIGH_VOL_RATIO
        high_vol_adjust = 0.6
        mid_vol_ratio = 1.2
        mid_vol_adjust = 0.3
        if ratio >= high_vol_ratio:
            return base + high_vol_adjust
        elif ratio >= mid_vol_ratio:
            return base + mid_vol_adjust
        else:
            return base
    except Exception:
        return base

def get_adaptive_tsl_step(current_atr, avg_atr, default_step=None):
    """Determines the TSL step size (in R units) based on volatility."""
    if default_step is None:
        default_step = ADAPTIVE_TSL_DEFAULT_STEP_R
    high_vol_ratio = ADAPTIVE_TSL_HIGH_VOL_RATIO
    high_vol_step = ADAPTIVE_TSL_HIGH_VOL_STEP_R
    low_vol_ratio = ADAPTIVE_TSL_LOW_VOL_RATIO
    low_vol_step = ADAPTIVE_TSL_LOW_VOL_STEP_R

    current_atr_num = pd.to_numeric(current_atr, errors='coerce')
    avg_atr_num = pd.to_numeric(avg_atr, errors='coerce')

    if pd.isna(current_atr_num) or pd.isna(avg_atr_num) or np.isinf(current_atr_num) or np.isinf(avg_atr_num) or avg_atr_num < 1e-9:
        return default_step
    try:
        ratio = current_atr_num / avg_atr_num
        if ratio > high_vol_ratio:
            return high_vol_step
        elif ratio < low_vol_ratio:
            return low_vol_step
        else:
            return default_step
    except Exception:
        return default_step

# <<< [Patch] MODIFIED v4.8.8 (Patch 11): Renamed and simplified to only handle TSL >>>
def update_tsl_only(order, current_high, current_low, current_atr, avg_atr, atr_multiplier=1.5):
    """
    Updates SL for trailing stop loss (TSL) logic based on price movement and ATR.
    This function assumes TSL activation logic is handled elsewhere.
    Returns the order and a boolean indicating if the SL was updated.
    """
    sl_updated_by_tsl = False # <<< Patch 20: Added return flag
    if order is None: return order, sl_updated_by_tsl
    if not order.get('tsl_activated', False): # Only run if TSL is already active
        logging.debug(f"      [TSL Update] Skipping TSL update for order {order.get('entry_time')}: TSL not activated.")
        return order, sl_updated_by_tsl

    atr_val = pd.to_numeric(current_atr, errors='coerce')
    if pd.isna(atr_val) or atr_val <= 0:
        logging.warning(f"   (Warning) update_tsl_only: Invalid ATR ({atr_val}) for order {order.get('entry_time')}. Skipping TSL update.")
        return order, sl_updated_by_tsl

    current_sl = pd.to_numeric(order.get('sl_price'), errors='coerce')
    if pd.isna(current_sl):
        logging.warning(f"   (Warning) update_tsl_only: Invalid current SL price ({order.get('sl_price')}) for order {order.get('entry_time')}. Skipping TSL update.")
        return order, sl_updated_by_tsl

    # Update peak/trough since TSL activation
    if order.get('side') == 'BUY':
        order['peak_since_tsl_activation'] = max(order.get('peak_since_tsl_activation', order.get('entry_price')), current_high)
        peak_price = order['peak_since_tsl_activation']
        potential_new_sl = peak_price - atr_val * atr_multiplier
        logging.debug(f"      [TSL Calc] Buy Order {order.get('entry_time')}: Peak={peak_price:.5f}, ATR={atr_val:.3f}, Potential New SL={potential_new_sl:.5f}, Current SL={current_sl:.5f}")
        if potential_new_sl > current_sl:
            logging.info(f"      [TSL Update] Buy Order {order.get('entry_time')}: New SL={potential_new_sl:.5f} (Old SL={current_sl:.5f})")
            order['sl_price'] = potential_new_sl
            sl_updated_by_tsl = True # <<< Patch 20: Set flag
    elif order.get('side') == 'SELL':
        order['trough_since_tsl_activation'] = min(order.get('trough_since_tsl_activation', order.get('entry_price')), current_low)
        trough_price = order['trough_since_tsl_activation']
        potential_new_sl = trough_price + atr_val * atr_multiplier
        logging.debug(f"      [TSL Calc] Sell Order {order.get('entry_time')}: Trough={trough_price:.5f}, ATR={atr_val:.3f}, Potential New SL={potential_new_sl:.5f}, Current SL={current_sl:.5f}")
        if potential_new_sl < current_sl:
            logging.info(f"      [TSL Update] Sell Order {order.get('entry_time')}: New SL={potential_new_sl:.5f} (Old SL={current_sl:.5f})")
            order['sl_price'] = potential_new_sl
            sl_updated_by_tsl = True # <<< Patch 20: Set flag

    return order, sl_updated_by_tsl # <<< Patch 20: Return flag
# <<< End of [Patch] MODIFIED v4.8.8 (Patch 11) >>>

def update_trailing_tp2(order, atr, multiplier):
    """
    Updates TP2 target only after TP1 has been reached.
    (v4.8.8 Patch 7: Applied user prompt logic)
    """
    if order is None: return order

    if order.get('reached_tp1', False) and order.get('tp2_price') is None: # Original logic was tp2_price is None, which is correct to set it once
        entry = pd.to_numeric(order.get('entry_price'), errors='coerce')
        atr_val = pd.to_numeric(atr, errors='coerce')
        multiplier_val = pd.to_numeric(multiplier, errors='coerce')

        if pd.notna(entry) and pd.notna(atr_val) and pd.notna(multiplier_val) and atr_val > 0:
            if order.get('side') == 'BUY':
                tp2 = entry + atr_val * multiplier_val
                logging.info(f"[Patch] TP2 target set to {tp2:.5f} for order {order.get('entry_time')}")
                order['tp_price'] = tp2 # Update the main tp_price for exit check
                order['tp2_price'] = tp2 # Also store in tp2_price for clarity if needed elsewhere
            elif order.get('side') == 'SELL':
                tp2 = entry - atr_val * multiplier_val
                logging.info(f"[Patch] TP2 target set to {tp2:.5f} for order {order.get('entry_time')}")
                order['tp_price'] = tp2
                order['tp2_price'] = tp2
        else:
            logging.warning(f"   (Warning) Cannot set TP2 for order {order.get('entry_time')}: Invalid input values (entry={entry}, atr={atr_val}, mult={multiplier_val}).")
    return order


def spike_guard_london(row, session, consecutive_losses):
    """
    Applies spike guard filter, primarily for London session.
    (v4.8.8 Patch 5: Re-verified logic)
    """
    if not ENABLE_SPIKE_GUARD:
        return True
    if not isinstance(session, str) or "London" not in session:
        return True

    spike_score_val = pd.to_numeric(row.get("spike_score"), errors='coerce')
    if pd.notna(spike_score_val) and spike_score_val > 0.85:
        logging.debug(f"      (Spike Guard Filtered) Reason: London Session & High Spike Score ({spike_score_val:.2f} > 0.85)")
        return False

    adx_val = pd.to_numeric(row.get("ADX"), errors='coerce')
    wick_ratio_val = pd.to_numeric(row.get("Wick_Ratio"), errors='coerce')
    vol_index_val = pd.to_numeric(row.get("Volatility_Index"), errors='coerce')
    candle_body_val = pd.to_numeric(row.get("Candle_Body"), errors='coerce')
    candle_range_val = pd.to_numeric(row.get("Candle_Range"), errors='coerce')
    gain_val = pd.to_numeric(row.get("Gain"), errors='coerce')
    atr_val = pd.to_numeric(row.get("ATR_14"), errors='coerce')

    if any(pd.isna(v) for v in [adx_val, wick_ratio_val, vol_index_val, candle_body_val, candle_range_val, gain_val, atr_val]):
        return True

    safe_candle_range_val = max(candle_range_val, 1e-9)

    if adx_val < 20 and wick_ratio_val > 0.7 and vol_index_val < 0.8:
        logging.debug(f"      (Spike Guard Filtered) Reason: Low ADX({adx_val:.1f}), High Wick({wick_ratio_val:.2f}), Low Vol({vol_index_val:.2f})")
        return False

    try:
        body_ratio = candle_body_val / safe_candle_range_val
        if body_ratio < 0.07:
            logging.debug(f"      (Spike Guard Filtered) Reason: Low Body Ratio ({body_ratio:.3f})")
            return False
    except ZeroDivisionError:
        logging.warning("      (Spike Guard) ZeroDivisionError calculating body_ratio.")
        return False

    if gain_val > 3 and atr_val > 4 and (candle_body_val / safe_candle_range_val) > 0.3:
        logging.debug("      (Spike Guard Allowed) Reason: Strong directional move override.")
        return True

    return True

def is_entry_allowed(row, session, consecutive_losses, signal_score_threshold=None):
    """Checks if entry is allowed based on filters."""
    if signal_score_threshold is None:
        global MIN_SIGNAL_SCORE_ENTRY
        signal_score_threshold = MIN_SIGNAL_SCORE_ENTRY

    if not spike_guard_london(row, session, consecutive_losses):
        return False, "SPIKE_GUARD_LONDON"

    signal_score = pd.to_numeric(row.get("Signal_Score"), errors='coerce')
    if pd.isna(signal_score):
        return False, "INVALID_SIGNAL_SCORE (NaN)"
    if abs(signal_score) < signal_score_threshold:
        return False, f"LOW_SIGNAL_SCORE ({signal_score:.2f}<{signal_score_threshold})"

    return True, "ALLOWED"

def adjust_lot_recovery_mode(base_lot, consecutive_losses):
    """Adjusts lot size if in recovery mode."""
    global RECOVERY_MODE_CONSECUTIVE_LOSSES, RECOVERY_MODE_LOT_MULTIPLIER, MIN_LOT_SIZE
    if consecutive_losses >= RECOVERY_MODE_CONSECUTIVE_LOSSES:
        adjusted_lot = max(base_lot * RECOVERY_MODE_LOT_MULTIPLIER, MIN_LOT_SIZE)
        if not math.isclose(adjusted_lot, base_lot):
            logging.info(f"      (Recovery Mode Active) Losses: {consecutive_losses}. Lot adjusted: {base_lot:.2f} -> {adjusted_lot:.2f}")
        return adjusted_lot, "recovery"
    else:
        return base_lot, "normal"

def calculate_aggressive_lot(equity, max_lot=None):
    """Calculates lot size based on aggressive equity tiers."""
    if max_lot is None:
        global MAX_LOT_SIZE
        max_lot = MAX_LOT_SIZE
    global MIN_LOT_SIZE

    if equity < 100: lot = 0.01
    elif equity < 500: lot = 0.05
    elif equity < 1000: lot = 0.10
    elif equity < 3000: lot = 0.30
    elif equity < 5000: lot = 0.50
    elif equity < 8000: lot = 1.00
    else: lot = 2.00
    final_lot = round(min(lot, max_lot), 2)
    final_lot = max(final_lot, MIN_LOT_SIZE)
    return final_lot

def calculate_lot_size_fixed_risk(equity, risk_per_trade, sl_delta_price, point_value=None, min_lot=None, max_lot=None):
    """Calculates lot size based on fixed fractional risk."""
    if point_value is None: global POINT_VALUE; point_value = POINT_VALUE
    if min_lot is None: global MIN_LOT_SIZE; min_lot = MIN_LOT_SIZE
    if max_lot is None: global MAX_LOT_SIZE; max_lot = MAX_LOT_SIZE

    equity_num = pd.to_numeric(equity, errors='coerce')
    risk_num = pd.to_numeric(risk_per_trade, errors='coerce')
    sl_delta_num = pd.to_numeric(sl_delta_price, errors='coerce')

    if pd.isna(equity_num) or np.isinf(equity_num) or equity_num <= 0 or \
       pd.isna(risk_num) or np.isinf(risk_num) or risk_num <= 0 or \
       pd.isna(sl_delta_num) or np.isinf(sl_delta_num) or sl_delta_num <= 1e-9:
        return min_lot

    try:
        risk_amount_usd = equity_num * risk_num
        sl_points = sl_delta_num * 10.0
        risk_per_001_lot = sl_points * point_value
        if risk_per_001_lot <= 1e-9:
            return min_lot

        raw_lot_units = risk_amount_usd / risk_per_001_lot
        lot_size = raw_lot_units * 0.01
        lot_size = round(lot_size, 2)
        lot_size = max(min_lot, lot_size)
        lot_size = min(max_lot, lot_size)
        return lot_size
    except Exception:
        return min_lot

def adjust_lot_tp2_boost(trade_history, base_lot=0.01):
    """Increases lot size slightly after a streak of full TPs."""
    global MIN_LOT_SIZE
    boost_factor = 1.2
    streak_length = 2
    if len(trade_history) < streak_length:
        return base_lot
    full_trade_reasons = [str(reason) for reason in trade_history if not str(reason).startswith("Partial")]
    if len(full_trade_reasons) >= streak_length and all(t.upper() == "TP" for t in full_trade_reasons[-streak_length:]):
        boosted_lot = round(base_lot * boost_factor, 2)
        final_lot = max(boosted_lot, MIN_LOT_SIZE)
        if final_lot > base_lot:
            logging.info(f"      (TP Boost) Last {streak_length} full trades were TP. Lot boosted: {base_lot:.2f} -> {final_lot:.2f}")
        return final_lot
    return base_lot

def calculate_lot_by_fund_mode(mm_mode, risk_pct, equity, atr_at_entry, sl_delta_price):
    """Calculates base lot size based on the fund's money management mode."""
    global MAX_LOT_SIZE, MIN_LOT_SIZE
    base_lot = MIN_LOT_SIZE

    if mm_mode in ['conservative', 'mirror']:
        base_lot = calculate_lot_size_fixed_risk(equity, risk_pct, sl_delta_price)
    elif mm_mode == 'balanced':
        base_lot = calculate_aggressive_lot(equity)
    elif mm_mode == 'high_freq':
        if equity < 100: base_lot = 0.01
        elif equity < 500: base_lot = 0.02
        elif equity < 1000: base_lot = 0.03
        else: base_lot = 0.05
    elif mm_mode == 'spike_only':
        atr_threshold = 2.0
        atr_at_entry_num = pd.to_numeric(atr_at_entry, errors='coerce')
        if pd.notna(atr_at_entry_num) and atr_at_entry_num > atr_threshold:
            base_lot = calculate_aggressive_lot(equity)
        else:
            base_lot = calculate_lot_size_fixed_risk(equity, risk_pct, sl_delta_price)
    else:
        base_lot = calculate_aggressive_lot(equity)

    final_lot = round(min(base_lot, MAX_LOT_SIZE), 2)
    final_lot = max(final_lot, MIN_LOT_SIZE)
    return final_lot

# <<< [Patch] MODIFIED v4.8.8 (Patch 26.5.1): Applying [PATCH B - Unified] for logging fix. >>>
def check_main_exit_conditions(order, row, current_bar_index, now_timestamp):
    """
    [PATCH B - Unified] Checks exit conditions for an order in strict priority: BE-SL -> SL -> TP -> MaxBars.
    Uses tolerance for price checks. Renamed from _check_order_exit_conditions.

    Args:
        order (dict): The active order dictionary.
        row (pd.Series): The current market data row (OHLC).
        current_bar_index (int): The index of the current bar.
        now_timestamp (pd.Timestamp): The timestamp of the current bar.

    Returns:
        tuple: (order_closed_this_bar, exit_price, close_reason, close_timestamp)
    """
    global MAX_HOLDING_BARS

    order_closed_this_bar = False
    exit_price_final = np.nan
    close_reason_final = "UNKNOWN_EXIT" # Default if no condition met
    close_timestamp_final = now_timestamp

    side = order.get("side")
    sl_price_order = pd.to_numeric(order.get("sl_price"), errors='coerce')
    tp_price_order = pd.to_numeric(order.get("tp_price"), errors='coerce')
    entry_price_order = pd.to_numeric(order.get("entry_price"), errors='coerce')

    current_high = pd.to_numeric(row.get("High"), errors='coerce')
    current_low = pd.to_numeric(row.get("Low"), errors='coerce')
    current_close = pd.to_numeric(row.get("Close"), errors='coerce')
    be_triggered = order.get('be_triggered', False)
    entry_bar_count_order = order.get("entry_bar_count")
    entry_time_log = order.get('entry_time', 'N/A') # For logging

    price_tolerance = 0.05

    # <<< [Patch B - Unified] Applied to logging in check_main_exit_conditions >>>
    sl_text = f"{sl_price_order:.5f}" if pd.notna(sl_price_order) else "NaN"
    tp_text = f"{tp_price_order:.5f}" if pd.notna(tp_price_order) else "NaN"
    logging.debug(
        f"            [Exit Check V2.1] Order {entry_time_log} "
        f"Side: {side}, SL: {sl_text}, TP: {tp_text}, BE: {be_triggered}"
    )
    # <<< End of [Patch B - Unified] >>>
    logging.debug(f"            [Exit Check V2.1] Bar Prices: H={current_high:.5f}, L={current_low:.5f}, C={current_close:.5f}")

    if be_triggered and pd.notna(sl_price_order) and pd.notna(entry_price_order) and math.isclose(sl_price_order, entry_price_order, abs_tol=price_tolerance):
        if side == 'BUY' and (current_low <= sl_price_order + price_tolerance):
            order_closed_this_bar = True; close_reason_final = 'BE-SL'; exit_price_final = sl_price_order
            logging.info(f"               [Patch B Check] BE-SL HIT (BUY). Order {entry_time_log}. Exit Price: {exit_price_final:.5f}")
        elif side == 'SELL' and (current_high >= sl_price_order - price_tolerance):
            order_closed_this_bar = True; close_reason_final = 'BE-SL'; exit_price_final = sl_price_order
            logging.info(f"               [Patch B Check] BE-SL HIT (SELL). Order {entry_time_log}. Exit Price: {exit_price_final:.5f}")

    if not order_closed_this_bar and pd.notna(sl_price_order):
        if side == 'BUY' and (current_low <= sl_price_order + price_tolerance):
            order_closed_this_bar = True; close_reason_final = 'SL'; exit_price_final = sl_price_order
            logging.info(f"               [Patch B Check] SL HIT (BUY). Order {entry_time_log}. Exit Price: {exit_price_final:.5f}")
        elif side == 'SELL' and (current_high >= sl_price_order - price_tolerance):
            order_closed_this_bar = True; close_reason_final = 'SL'; exit_price_final = sl_price_order
            logging.info(f"               [Patch B Check] SL HIT (SELL). Order {entry_time_log}. Exit Price: {exit_price_final:.5f}")

    if not order_closed_this_bar and pd.notna(tp_price_order):
        if side == 'BUY' and (current_high >= tp_price_order - price_tolerance):
            order_closed_this_bar = True; close_reason_final = 'TP'; exit_price_final = tp_price_order
            logging.info(f"               [Patch B Check] TP HIT (BUY). Order {entry_time_log}. Exit Price: {exit_price_final:.5f}")
        elif side == 'SELL' and (current_low <= tp_price_order + price_tolerance):
            order_closed_this_bar = True; close_reason_final = 'TP'; exit_price_final = tp_price_order
            logging.info(f"               [Patch B Check] TP HIT (SELL). Order {entry_time_log}. Exit Price: {exit_price_final:.5f}")

    if not order_closed_this_bar:
        if entry_bar_count_order is not None:
            bars_held = current_bar_index - entry_bar_count_order
            logging.debug(f"            [Exit Check V2.1] MaxBars: Bars Held={bars_held}, Max={MAX_HOLDING_BARS}")
            if bars_held >= MAX_HOLDING_BARS:
                logging.info(f"      [Patch B Check] Max Holding Bars ({MAX_HOLDING_BARS}) reached for order {order.get('entry_time')} at {now_timestamp}.")
                if pd.notna(current_close):
                    exit_price_final = current_close; close_reason_final = f"MaxBars ({MAX_HOLDING_BARS})"; order_closed_this_bar = True
                else:
                    logging.warning(f"      (Warning) Order {order.get('entry_time')} hit MaxBars, but close price is NaN. Using SL if valid, else entry.")
                    exit_price_final = sl_price_order if pd.notna(sl_price_order) else entry_price_order
                    if pd.isna(exit_price_final): exit_price_final = 0; logging.error(f"          (Error) MaxBars Exit: SL and Entry price are NaN for order {order.get('entry_time')}. Using 0 as exit price.")
                    close_reason_final = f"MaxBars ({MAX_HOLDING_BARS})_CloseNaN"; order_closed_this_bar = True
        else:
            logging.warning(f"      (Warning) Cannot check MaxBars for order {order.get('entry_time')}: Missing 'entry_bar_count'.")

    return order_closed_this_bar, exit_price_final, close_reason_final, close_timestamp_final

# <<< [Patch] MODIFIED v4.8.8 (Patch 26.5.1): Applied [PATCH B - Unified] for logging. >>>
def _update_open_order_state(order, current_high, current_low, current_atr, avg_atr, now, base_be_r_thresh, fold_sl_multiplier_base, base_tp_multiplier_config, be_sl_counter, tsl_counter):
    """
    Updates the state (BE, TSL, TTP2) of an order that remains open in the current bar.
    Prioritizes BE trigger over TSL activation/update.
    """
    global DYNAMIC_BE_ATR_THRESHOLD_HIGH, DYNAMIC_BE_R_ADJUST_HIGH, ADAPTIVE_TSL_START_ATR_MULT

    be_triggered_this_bar = False
    tsl_updated_this_bar = False
    order_side = order.get("side")
    entry_price = pd.to_numeric(order.get("entry_price"), errors='coerce')
    original_sl_price = pd.to_numeric(order.get("original_sl_price"), errors='coerce')
    current_sl_price_in_order = pd.to_numeric(order.get("sl_price"), errors='coerce')
    atr_at_entry = pd.to_numeric(order.get("atr_at_entry"), errors='coerce')
    entry_time_log = order.get('entry_time', 'N/A') # For logging

    if not order.get("be_triggered", False):
        dynamic_be_r_threshold = base_be_r_thresh
        try:
            current_atr_for_be_calc = pd.to_numeric(atr_at_entry, errors='coerce')
            current_avg_atr_for_be_calc = pd.to_numeric(avg_atr, errors='coerce')
            if (pd.notna(current_atr_for_be_calc) and pd.notna(current_avg_atr_for_be_calc) and
                not np.isinf(current_atr_for_be_calc) and not np.isinf(current_avg_atr_for_be_calc) and
                current_avg_atr_for_be_calc > 1e-9 and
                (current_atr_for_be_calc / current_avg_atr_for_be_calc) > DYNAMIC_BE_ATR_THRESHOLD_HIGH):
                dynamic_be_r_threshold += DYNAMIC_BE_R_ADJUST_HIGH
                logging.debug(f"            [Patch BE] Dynamic BE Threshold adjusted to {dynamic_be_r_threshold:.2f}R due to high volatility (ATR@Entry/AvgATR > {DYNAMIC_BE_ATR_THRESHOLD_HIGH}).")
        except Exception as e_be_dyn:
            logging.warning(f"      (Warning) Error calculating dynamic BE threshold at {now}: {e_be_dyn}")

        if dynamic_be_r_threshold > 0:
            if pd.notna(entry_price) and pd.notna(original_sl_price):
                sl_delta_price_be = abs(entry_price - original_sl_price)
                if sl_delta_price_be > 1e-9:
                    be_trigger_price_diff = sl_delta_price_be * dynamic_be_r_threshold
                    be_trigger_price = entry_price + be_trigger_price_diff if order_side == "BUY" else entry_price - be_trigger_price_diff
                    trigger_hit = False
                    current_sl_text = f"{current_sl_price_in_order:.5f}" if pd.notna(current_sl_price_in_order) else "NaN"
                    logging.debug(f"            [BE Check] Order {entry_time_log}: High={current_high:.5f}, Low={current_low:.5f}, BE Trigger Price={be_trigger_price:.5f}, Current SL={current_sl_text}")
                    if order_side == "BUY" and current_high >= be_trigger_price: trigger_hit = True
                    elif order_side == "SELL" and current_low <= be_trigger_price: trigger_hit = True
                    if trigger_hit and not math.isclose(current_sl_price_in_order if pd.notna(current_sl_price_in_order) else -np.inf, entry_price if pd.notna(entry_price) else np.inf, rel_tol=1e-9, abs_tol=1e-9): # Check for NaN before math.isclose
                        logging.info(f"         [Patch BE] Breakeven Triggered for order {entry_time_log} at {now}. Moving SL from {current_sl_text} to {entry_price:.5f}")
                        order["sl_price"] = entry_price; order["be_triggered"] = True; order["be_triggered_time"] = now; be_sl_counter += 1; be_triggered_this_bar = True
                else: logging.debug(f"            Skipping BE check for order {entry_time_log}: Invalid SL delta ({sl_delta_price_be}) from original_sl_price.")
            else: logging.debug(f"            Skipping BE check for order {entry_time_log}: Invalid entry or original_sl_price.")

    if not be_triggered_this_bar:
        if not order.get("tsl_activated", False) and pd.notna(atr_at_entry) and atr_at_entry > 1e-9:
            tsl_activation_price_diff = ADAPTIVE_TSL_START_ATR_MULT * atr_at_entry
            tsl_activation_price = entry_price + tsl_activation_price_diff if order_side == "BUY" else entry_price - tsl_activation_price_diff
            logging.debug(f"            [TSL Activation Check] Order {entry_time_log}: High={current_high:.5f}, Low={current_low:.5f}, Activation Price={tsl_activation_price:.5f}")
            if (order_side == "BUY" and current_high >= tsl_activation_price) or (order_side == "SELL" and current_low <= tsl_activation_price):
                logging.info(f"         [Patch TSL] Trailing Stop Loss (TSL) ACTIVATED for order {entry_time_log} at {now}.")
                order["tsl_activated"] = True
                if order_side == "BUY": order["peak_since_tsl_activation"] = current_high
                else: order["trough_since_tsl_activation"] = current_low
        if order.get("tsl_activated"):
            tsl_atr_mult_fixed = 1.5
            sl_before_tsl_text = f"{order.get('sl_price'):.5f}" if pd.notna(order.get('sl_price')) else "NaN"
            logging.debug(f"            [TSL Update Call] Order {entry_time_log}. SL before={sl_before_tsl_text}, ATR Mult={tsl_atr_mult_fixed:.2f}")
            order, sl_updated_flag = update_tsl_only(order, current_high, current_low, current_atr, avg_atr, atr_multiplier=tsl_atr_mult_fixed)
            if sl_updated_flag: tsl_updated_this_bar = True; tsl_counter += 1
            new_sl_price_after_tsl_val = pd.to_numeric(order.get("sl_price"), errors='coerce')
            sl_after_tsl_text = f"{new_sl_price_after_tsl_val:.5f}" if pd.notna(new_sl_price_after_tsl_val) else "NaN"
            logging.debug(f"            Order {entry_time_log} after update_tsl_only. SL after={sl_after_tsl_text}")
    else:
        logging.debug(f"            Skipping TSL checks for order {entry_time_log} because BE was triggered in this bar.")

    tp2_mult = dynamic_tp2_multiplier(current_atr, avg_atr, base=base_tp_multiplier_config)
    tp_price_val_before = order.get('tp_price')
    tp_before_str = f"{tp_price_val_before:.5f}" if pd.notna(tp_price_val_before) else "NaN"
    logging.debug(
        f"            [TTP2 Update Call] Order {entry_time_log}. "
        f"TP before={tp_before_str}, Dyn TP2 Mult={tp2_mult:.2f}"
    )
    order = update_trailing_tp2(order, current_atr, tp2_mult)
    tp_price_val_after = order.get('tp_price')
    tp_after_str = f"{tp_price_val_after:.5f}" if pd.notna(tp_price_val_after) else "NaN"
    logging.debug(f"            Order {entry_time_log} after update_trailing_tp2. TP after={tp_after_str}")

    return order, be_triggered_this_bar, tsl_updated_this_bar, be_sl_counter, tsl_counter


# --- Backtesting Simulation Engine ---
# <<< [Patch] MODIFIED v4.8.8 (Patch 26.5.1): Applied [PATCH C - Unified] for error handling and logging fix. >>>
def run_backtest_simulation_v34(
    df_m1_segment_pd,
    label,
    initial_capital_segment,
    side="BUY",
    fund_profile=None,
    fold_config=None,
    available_models=None,
    model_switcher_func=None,
    pattern_label_map=None,
    meta_min_proba_thresh_override=None,
    current_fold_index=None,
    enable_partial_tp=ENABLE_PARTIAL_TP,
    partial_tp_levels=PARTIAL_TP_LEVELS,
    partial_tp_move_sl_to_entry=PARTIAL_TP_MOVE_SL_TO_ENTRY,
    enable_kill_switch=ENABLE_KILL_SWITCH,
    kill_switch_max_dd_threshold=KILL_SWITCH_MAX_DD_THRESHOLD,
    kill_switch_consecutive_losses_config=KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD,
    recovery_mode_consecutive_losses_config=RECOVERY_MODE_CONSECUTIVE_LOSSES,
    min_equity_threshold_pct=min_equity_threshold_pct,
    initial_kill_switch_state=False,
    initial_consecutive_losses=0,
):
    """
    Runs the core backtesting simulation loop for a single fold, side, and fund profile.
    (v4.8.8 Patch 26.5.1: Unified error handling, logging, and exit logic fixes)
    """
    global meta_model_type_used, meta_meta_model_type_used, USE_REENTRY, REENTRY_COOLDOWN_BARS, TIMEFRAME_MINUTES_M1, POINT_VALUE, MAX_CONCURRENT_ORDERS, MAX_HOLDING_BARS, COMMISSION_PER_001_LOT, SPREAD_POINTS, MIN_SLIPPAGE_POINTS, MAX_SLIPPAGE_POINTS, MAX_DRAWDOWN_THRESHOLD, ENABLE_FORCED_ENTRY, FORCED_ENTRY_BAR_THRESHOLD, FORCED_ENTRY_MIN_SIGNAL_SCORE, FORCED_ENTRY_LOOKBACK_PERIOD, FORCED_ENTRY_CHECK_MARKET_COND, FORCED_ENTRY_MAX_ATR_MULT, FORCED_ENTRY_MIN_GAIN_Z_ABS, FORCED_ENTRY_ALLOWED_REGIMES, FE_ML_FILTER_THRESHOLD, forced_entry_max_consecutive_losses, OUTPUT_DIR, USE_META_CLASSIFIER, BASE_BE_SL_R_THRESHOLD, DYNAMIC_BE_ATR_THRESHOLD_HIGH, DYNAMIC_BE_R_ADJUST_HIGH, META_MIN_PROBA_THRESH, REENTRY_MIN_PROBA_THRESH

    meta_proba_tp_for_log = np.nan; meta2_proba_tp_for_log = np.nan; meta_proba_tp_for_fe_log = np.nan; total_ib_lot_accumulator = 0.0
    equity = initial_capital_segment; peak_equity = initial_capital_segment; max_drawdown_pct = 0.0
    active_orders = []; trade_log = []; blocked_order_log = []
    start_time_equity = df_m1_segment_pd.index[0] if not df_m1_segment_pd.empty else pd.Timestamp.now(tz='UTC')
    equity_history = {start_time_equity: initial_capital_segment}
    total_commission_paid = 0.0; total_slippage_loss = 0.0; total_spread_cost = 0.0
    orders_blocked_by_drawdown = 0; orders_blocked_by_cooldown = 0; orders_lot_scaled = 0
    be_sl_triggered_count_run = 0; tsl_triggered_count_run = 0; orders_skipped_ml_l1 = 0; orders_skipped_ml_l2 = 0
    reentry_trades_opened = 0; forced_entry_trades_opened = 0
    min_ts = pd.Timestamp.min.tz_localize('UTC') if not df_m1_segment_pd.empty and df_m1_segment_pd.index.tz is not None else pd.Timestamp.min
    last_trade_cooldown_end_time = defaultdict(lambda: min_ts); last_tp_time = defaultdict(lambda: min_ts)
    bars_since_last_trade = 0; kill_switch_activated = initial_kill_switch_state; consecutive_losses = initial_consecutive_losses
    forced_entry_consecutive_losses = 0; forced_entry_temporarily_disabled = False; last_n_full_trade_pnls = []
    SOFT_COOLDOWN_LOOKBACK = 10; SOFT_COOLDOWN_LOSS_COUNT = 3; kill_switch_trigger_time = pd.NaT
    current_risk_mode = "normal"; trade_history_list = []
    error_in_loop = False

    if not isinstance(df_m1_segment_pd, pd.DataFrame): logging.error(f"   (Error) Invalid input: df_m1_segment_pd is not a DataFrame for {label}."); run_summary_error = {"error_in_loop": True, "total_commission": 0, "total_spread": 0, "total_slippage": 0}; return pd.DataFrame(), pd.DataFrame(), initial_capital_segment, equity_history, 0.0, run_summary_error, blocked_order_log, "N/A", "N/A", initial_kill_switch_state, initial_consecutive_losses, 0.0
    if df_m1_segment_pd.empty: logging.warning(f"   (Warning) Input DataFrame is empty for {label}. Skipping simulation."); run_summary_error = {"error_in_loop": False, "total_commission": 0, "total_spread": 0, "total_slippage": 0}; return df_m1_segment_pd, pd.DataFrame(), initial_capital_segment, equity_history, 0.0, run_summary_error, blocked_order_log, "N/A", "N/A", initial_kill_switch_state, initial_consecutive_losses, 0.0
    if fold_config is None or not isinstance(fold_config, dict): logging.warning(f"   (Warning) Invalid fold_config provided for {label}. Using empty dict."); fold_config = {}
    if current_fold_index is None or not isinstance(current_fold_index, int) or current_fold_index < 0: logging.error(f"   (Error) Invalid current_fold_index: {current_fold_index} for {label}."); run_summary_error = {"error_in_loop": True, "total_commission": 0, "total_spread": 0, "total_slippage": 0}; return df_m1_segment_pd, pd.DataFrame(), initial_capital_segment, equity_history, 0.0, run_summary_error, blocked_order_log, "N/A", "N/A", initial_kill_switch_state, initial_consecutive_losses, 0.0
    if side not in ["BUY", "SELL"]: logging.error(f"   (Error) Invalid side: {side} for {label}."); run_summary_error = {"error_in_loop": True, "total_commission": 0, "total_spread": 0, "total_slippage": 0}; return df_m1_segment_pd, pd.DataFrame(), initial_capital_segment, equity_history, 0.0, run_summary_error, blocked_order_log, "N/A", "N/A", initial_kill_switch_state, initial_consecutive_losses, 0.0
    if fund_profile is None or not isinstance(fund_profile, dict) or "mm_mode" not in fund_profile or "risk" not in fund_profile: logging.warning(f"   (Warning) Invalid fund_profile provided for {label}. Using default."); fund_profile = {"name": DEFAULT_FUND_NAME, "risk": DEFAULT_RISK_PER_TRADE, "mm_mode": "balanced"}
    if USE_META_CLASSIFIER:
        if model_switcher_func is not None and not callable(model_switcher_func): logging.error(f"   (Error) model_switcher_func is not callable for {label}."); run_summary_error = {"error_in_loop": True, "total_commission": 0, "total_spread": 0, "total_slippage": 0}; return df_m1_segment_pd, pd.DataFrame(), initial_capital_segment, equity_history, 0.0, run_summary_error, blocked_order_log, "N/A", "N/A", initial_kill_switch_state, initial_consecutive_losses, 0.0
        if available_models is None or not isinstance(available_models, dict) or not available_models:
            if model_switcher_func is not None: logging.error(f"   (Error) available_models is missing or invalid for {label} when model_switcher_func is provided."); run_summary_error = {"error_in_loop": True, "total_commission": 0, "total_spread": 0, "total_slippage": 0}; return df_m1_segment_pd, pd.DataFrame(), initial_capital_segment, equity_history, 0.0, run_summary_error, blocked_order_log, "N/A", "N/A", initial_kill_switch_state, initial_consecutive_losses, 0.0
        elif model_switcher_func is not None and ('main' not in available_models or available_models.get('main') is None or 'model' not in available_models['main'] or available_models['main'].get('model') is None): logging.error(f"   (Error) Main model is missing or invalid in available_models for {label} when model_switcher_func is provided."); run_summary_error = {"error_in_loop": True, "total_commission": 0, "total_spread": 0, "total_slippage": 0}; return df_m1_segment_pd, pd.DataFrame(), initial_capital_segment, equity_history, 0.0, run_summary_error, blocked_order_log, "N/A", "N/A", initial_kill_switch_state, initial_consecutive_losses, 0.0

    local_enable_partial_tp = enable_partial_tp; local_partial_tp_levels = []
    if local_enable_partial_tp:
        if not isinstance(partial_tp_levels, list) or not partial_tp_levels: local_enable_partial_tp = False
        else:
            valid_levels = True
            for level_idx, level in enumerate(partial_tp_levels):
                if not isinstance(level, dict) or "r_multiple" not in level or "close_pct" not in level: valid_levels = False; break
                if not isinstance(level["r_multiple"], (int, float)) or level["r_multiple"] <= 0: valid_levels = False; break
                if not isinstance(level["close_pct"], (int, float)) or not (0 < level["close_pct"] <= 1.0): valid_levels = False; break
            if not valid_levels: local_enable_partial_tp = False
            else: local_partial_tp_levels = sorted(partial_tp_levels, key=lambda x: x['r_multiple']);
            if sum(level['close_pct'] for level in local_partial_tp_levels) > 1.0 + 1e-9: logging.warning(f"  (Warning) Sim {label}: Sum of partial_tp_levels close_pct > 1.0.")

    current_meta_threshold_l1 = META_MIN_PROBA_THRESH
    if isinstance(meta_min_proba_thresh_override, dict): current_meta_threshold_l1 = meta_min_proba_thresh_override.get(current_fold_index, META_MIN_PROBA_THRESH)
    elif isinstance(meta_min_proba_thresh_override, (float, int)): current_meta_threshold_l1 = float(meta_min_proba_thresh_override)
    current_reentry_threshold_l1 = REENTRY_MIN_PROBA_THRESH; sim_model_type_l1 = "Switcher" if USE_META_CLASSIFIER and callable(model_switcher_func) else ("ML_OFF" if not USE_META_CLASSIFIER else "NoSwitcher"); sim_model_type_l2 = "N/A"; meta_model_type_used = sim_model_type_l1; meta_meta_model_type_used = sim_model_type_l2
    label_suffix = f"_{label}"; logging.debug(f"Preparing DataFrame for simulation run: {label}")
    result_cols = ["Lot_Size", "Order_Opened", "Order_Closed_Time", "PnL_Realized_USD", "Commission_USD", "Spread_Cost_USD", "Slippage_USD", "Equity_Realistic", "Active_Order_Count", "Max_Drawdown_At_Point", "Exit_Reason_Actual", "Exit_Price_Actual", "PnL_Points_Actual", "M15_Trend_Zone", "M1_Entry_Signal", "Signal_Score", "Trade_Reason", "Session", "BE_Triggered_Time", "Is_Reentry", "Meta_Proba_TP", "Meta2_Proba_TP", "Forced_Entry", "Entry_Price_Actual", "SL_Price_Actual", "TP_Price_Actual", "ATR_At_Entry", "Equity_Before_Open", "Entry_Gain_Z", "Entry_MACD_Smooth", "Entry_Candle_Ratio", "Entry_ADX", "Entry_Volatility_Index", "Trade_Tag", "Risk_Mode", "Active_Model", "Model_Confidence"]
    df_sim = df_m1_segment_pd.copy()
    for col_base in result_cols:
        col_name = f"{col_base}{label_suffix}"
        if col_name not in df_sim.columns:
            if "Time" in col_base: default_val = pd.NaT
            elif any(x in col_base for x in ["Opened", "Reentry", "Forced_Entry"]): default_val = False
            elif any(x in col_base for x in ["Zone", "Signal", "Reason", "Session", "Tag", "Mode", "Model"]): default_val = "NONE"
            elif any(x in col_base for x in ["Score", "Count"]): default_val = 0
            elif any(x in col_base for x in ["Proba", "Price", "ATR", "Gain_Z", "MACD", "Ratio", "ADX", "Volatility", "Confidence"]): default_val = np.nan
            else: default_val = 0.0
            df_sim[col_name] = default_val
    logging.debug(f"Result columns initialized for suffix '{label_suffix}'.")
    base_cfg = ENTRY_CONFIG_PER_FOLD.get(current_fold_index, ENTRY_CONFIG_PER_FOLD.get(0, {})); fold_sl_multiplier_base = fold_config.get("sl_multiplier", base_cfg.get("sl_multiplier", 2.0)); logging.info(f"   [Patch B Check] Using SL Multiplier: {fold_sl_multiplier_base} for Fold {current_fold_index+1} (from fold_config or base_cfg)")
    base_be_r_thresh = BASE_BE_SL_R_THRESHOLD; base_tp_multiplier_config = BASE_TP_MULTIPLIER; local_forced_entry_min_gain_z_abs = FORCED_ENTRY_MIN_GAIN_Z_ABS
    ignore_rsi_scoring = fold_config.get('ignore_rsi_scoring', False); use_gain_based_exit = fold_config.get('use_gain_based_exit', False); drift_override_active = ignore_rsi_scoring or use_gain_based_exit; drift_override_reason = ""
    if ignore_rsi_scoring: drift_override_reason += "RSI_Drift "
    if use_gain_based_exit: drift_override_reason += "ATR_Drift "
    drift_override_reason = drift_override_reason.strip()
    ptp_display = f"{local_partial_tp_levels[0]['r_multiple']:.1f}R ({local_partial_tp_levels[0]['close_pct']*100:.0f}%)" if local_enable_partial_tp and local_partial_tp_levels else 'N/A'
    start_log_msg = (f"  (Starting) Backtest: {label} ({side}), Fold: {current_fold_index+1}, Capital:${initial_capital_segment:.2f}, MM Mode:'{fund_profile.get('mm_mode', 'N/A')}'(Risk:{fund_profile.get('risk', np.nan):.3f}), SLx:{fold_sl_multiplier_base}(ATR), BE@R:{base_be_r_thresh}(Dyn), TP1@R:1.0, TP2@R:Dynamic({base_tp_multiplier_config} Base), PartialTP@R:{ptp_display}, TSL@ATR:{ADAPTIVE_TSL_START_ATR_MULT}, Meta:{sim_model_type_l1}(T={current_meta_threshold_l1:.2f}), ReEntry:{'ON' if USE_REENTRY else 'OFF'}(Cool:{REENTRY_COOLDOWN_BARS}b, T={current_reentry_threshold_l1:.2f}), ForcedEntry:{'ON' if ENABLE_FORCED_ENTRY else 'OFF'}(Bars:{FORCED_ENTRY_BAR_THRESHOLD}, Score>={FORCED_ENTRY_MIN_SIGNAL_SCORE:.2f}), PartialTP:{'ON' if local_enable_partial_tp else 'OFF'}(MoveSL:{partial_tp_move_sl_to_entry}), KillSwitch:{'ON' if enable_kill_switch else 'OFF'}(DD>{kill_switch_max_dd_threshold*100:.0f}%, Losses>{kill_switch_consecutive_losses_config}), SpikeGuard:{'ON' if ENABLE_SPIKE_GUARD else 'OFF'}, RecoveryMM:{'ON' if RECOVERY_MODE_CONSECUTIVE_LOSSES > 0 else 'OFF'}(Losses>={recovery_mode_consecutive_losses_config}, LotMult={RECOVERY_MODE_LOT_MULTIPLIER}), MM Boost: ON, DriftOverride: {'ACTIVE (' + drift_override_reason + ')' if drift_override_active else 'Inactive'}")
    logging.info(start_log_msg)
    if use_gain_based_exit: logging.info("             >> Using Gain-Based Exit Logic (PLACEHOLDER) due to ATR Drift <<")
    required_cols_sim_base = ["Open", "High", "Low", "Close", "ATR_14_Shifted", "ATR_14", "ATR_14_Rolling_Avg", "Trend_Zone", "Gain_Z", "MACD_hist", "MACD_hist_smooth", "Candle_Speed", "Pattern_Label", "Entry_Long", "Entry_Short", "Trade_Tag", "Signal_Score", "Trade_Reason", "Volatility_Index", "ADX", "RSI", "Wick_Ratio", "Candle_Body", "Candle_Range", "Gain", 'cluster', 'spike_score', 'session']
    missing_sim_cols_base = [c for c in required_cols_sim_base if c not in df_sim.columns]
    if missing_sim_cols_base: logging.error(f"   (Error) Missing required columns in input DataFrame for {label}: {missing_sim_cols_base}"); run_summary_error = {"error_in_loop": True, "total_commission": 0, "total_spread": 0, "total_slippage": 0}; return df_sim, pd.DataFrame(), equity, equity_history, max_drawdown_pct, run_summary_error, blocked_order_log, sim_model_type_l1, sim_model_type_l2, kill_switch_activated, consecutive_losses, total_ib_lot_accumulator
    logging.info(f"Starting simulation loop for {label} ({len(df_sim)} bars)...")
    current_bar_index = 0; iterator_obj = df_sim.iterrows()
    if tqdm: iterator = tqdm(iterator_obj, total=df_sim.shape[0], desc=f"  Sim ({label}, {side})", leave=False, mininterval=2.0)
    else: iterator = iterator_obj
    run_summary = {}

    try:
        for idx, row in iterator:
            now = idx; equity_at_start_of_bar = equity; current_equity_change_this_bar = 0.0
            logging.debug(f"--- Bar {current_bar_index} ({idx}) --- Equity Start: {equity_at_start_of_bar:.2f}, Active Orders: {len(active_orders)}")
            current_open = pd.to_numeric(row.get("Open"), errors='coerce'); current_low = pd.to_numeric(row.get("Low"), errors='coerce'); current_high = pd.to_numeric(row.get("High"), errors='coerce'); current_close = pd.to_numeric(row.get("Close"), errors='coerce'); current_atr_shifted = pd.to_numeric(row.get("ATR_14_Shifted"), errors='coerce'); current_atr = pd.to_numeric(row.get("ATR_14"), errors='coerce'); current_avg_atr = pd.to_numeric(row.get("ATR_14_Rolling_Avg"), errors='coerce'); current_vol_index = pd.to_numeric(row.get("Volatility_Index"), errors='coerce'); current_macd_smooth = pd.to_numeric(row.get("MACD_hist_smooth"), errors='coerce'); current_signal_score = pd.to_numeric(row.get("Signal_Score"), errors='coerce'); current_rsi = pd.to_numeric(row.get("RSI"), errors='coerce'); current_gain_z = pd.to_numeric(row.get("Gain_Z"), errors='coerce'); current_trade_tag = row.get("Trade_Tag", "N/A"); session_tag = row.get("session", "Other")
            if any(pd.isna(p) or (isinstance(p, float) and np.isinf(p)) for p in [current_open, current_high, current_low, current_close]):
                logging.debug(f"   Skipping bar {idx} due to missing/invalid price data."); df_sim.loc[idx, f"Max_Drawdown_At_Point{label_suffix}"] = max_drawdown_pct; df_sim.loc[idx, f"Equity_Realistic{label_suffix}"] = equity; df_sim.loc[idx, f"Active_Order_Count{label_suffix}"] = len(active_orders); equity_history[now] = equity; current_bar_index += 1; continue
            next_active_orders = []; order_closed_this_bar_flag = False
            logging.debug(f"   Processing {len(active_orders)} active orders for bar {idx}...")
            # <<< [Patch C - Unified] Added try-except around order processing logic inside the bar loop >>>
            try:
                for order_index, order in enumerate(active_orders):
                    order_closed_this_bar = False; be_triggered_this_bar = False; exit_price = np.nan; close_reason = "UNKNOWN"; close_timestamp = now; order_entry_time = order.get("entry_time", "N/A")
                    logging.debug(f"      Processing Order {order_index+1} (Entry: {order_entry_time}). SL={order.get('sl_price')}, TP={order.get('tp_price')}, BE Trig={order.get('be_triggered', False)}")
                    order_side = order.get("side"); entry_price = pd.to_numeric(order.get("entry_price"), errors='coerce'); original_lot = pd.to_numeric(order.get("original_lot", order.get("lot")), errors='coerce'); current_lot = pd.to_numeric(order.get("lot", 0.0), errors='coerce'); original_sl_price = pd.to_numeric(order.get("original_sl_price"), errors='coerce'); current_sl_price_in_order = pd.to_numeric(order.get("sl_price"), errors='coerce'); current_tp_price_in_order = pd.to_numeric(order.get("tp_price"), errors='coerce'); atr_at_entry = pd.to_numeric(order.get("atr_at_entry"), errors='coerce'); atr_at_entry_log = order.get("atr_at_entry", np.nan); equity_before_open_log = order.get("equity_before_open", np.nan); entry_gain_z_log = order.get("entry_gain_z", np.nan); entry_macd_smooth_log = order.get("entry_macd_smooth", np.nan); entry_candle_ratio_log = order.get("entry_candle_ratio", np.nan); entry_adx_log = order.get("entry_adx", np.nan); entry_volatility_index_log = order.get("entry_volatility_index", np.nan); order_trade_tag = order.get("trade_tag", "N/A"); active_model_at_entry = order.get("active_model_at_entry", "N/A"); model_confidence_at_entry = order.get("model_confidence_at_entry", np.nan); risk_mode_at_entry_log = order.get("risk_mode_at_entry", "N/A")
                    if any(pd.isna(v) or (isinstance(v, float) and np.isinf(v)) for v in [entry_price, original_lot, current_lot, original_sl_price, current_sl_price_in_order, current_tp_price_in_order, atr_at_entry]): logging.error(f"      (Error) Skipping processing for order {order_entry_time} due to invalid numeric values in order dict."); next_active_orders.append(order); continue
                    if kill_switch_activated and not order.get("closed_by_killswitch", False): logging.warning(f"      Closing order {order_entry_time} due to Kill Switch at {now}."); order_closed_this_bar = True; close_reason = "Kill Switch"; exit_price = current_close; close_timestamp = now; order["closed_by_killswitch"] = True
                    partial_tp_processed_levels = order.get("partial_tp_processed_levels", set())
                    if not order_closed_this_bar and local_enable_partial_tp and current_lot >= MIN_LOT_SIZE and atr_at_entry > 1e-9:
                        sl_delta_price_ptp = atr_at_entry * fold_sl_multiplier_base
                        if sl_delta_price_ptp > 1e-9:
                            logging.debug(f"         Checking PTP for order {order_entry_time} (SL Delta: {sl_delta_price_ptp:.5f})...")
                            for level_idx, level_config in enumerate(local_partial_tp_levels):
                                if level_idx not in partial_tp_processed_levels:
                                    target_r = level_config["r_multiple"]; target_price_diff = sl_delta_price_ptp * target_r; partial_tp_price = entry_price + target_price_diff if side == "BUY" else entry_price - target_price_diff; partial_tp_hit = False; exit_price_ptp = np.nan; logging.debug(f"            Level {level_idx+1}: Target R={target_r:.1f}, Target Price={partial_tp_price:.5f}")
                                    if side == "BUY" and current_high >= partial_tp_price: partial_tp_hit = True; exit_price_ptp = partial_tp_price; order['reached_tp1'] = True; logging.debug(f"            PTP HIT (BUY): High={current_high:.5f} >= Target={partial_tp_price:.5f}")
                                    elif side == "SELL" and current_low <= partial_tp_price: partial_tp_hit = True; exit_price_ptp = partial_tp_price; order['reached_tp1'] = True; logging.debug(f"            PTP HIT (SELL): Low={current_low:.5f} <= Target={partial_tp_price:.5f}")
                                    if partial_tp_hit:
                                        logging.info(f"      Partial TP Level {level_idx+1} ({target_r:.1f}R) hit for order {order_entry_time} at {now}."); order_closed_this_bar_flag = True; close_pct = level_config["close_pct"]; lot_to_close = round(original_lot * close_pct, 2); lot_to_close = min(lot_to_close, current_lot); lot_to_close = max(lot_to_close, MIN_LOT_SIZE)
                                        if current_lot - lot_to_close < MIN_LOT_SIZE and current_lot - lot_to_close > 1e-9: lot_to_close = current_lot; logging.debug(f"         Adjusting partial close lot to {lot_to_close:.2f} to close full position.")
                                        if lot_to_close >= MIN_LOT_SIZE:
                                            close_reason_partial = f"Partial TP {level_idx+1} ({target_r:.1f}R)"; close_timestamp_partial = now; pnl_points_partial = (exit_price_ptp - entry_price) * 10.0 if side == "BUY" else (entry_price - exit_price_ptp) * 10.0; pnl_points_net_spread_partial = pnl_points_partial - SPREAD_POINTS; spread_cost_usd_partial = SPREAD_POINTS * (lot_to_close / MIN_LOT_SIZE) * POINT_VALUE; raw_pnl_usd_partial = pnl_points_net_spread_partial * (lot_to_close / MIN_LOT_SIZE) * POINT_VALUE; commission_usd_partial = (lot_to_close / MIN_LOT_SIZE) * COMMISSION_PER_001_LOT; slippage_points_partial = random.uniform(MIN_SLIPPAGE_POINTS, MAX_SLIPPAGE_POINTS); slippage_usd_partial = slippage_points_partial * (lot_to_close / MIN_LOT_SIZE) * POINT_VALUE; net_pnl_usd_partial = raw_pnl_usd_partial - commission_usd_partial + slippage_usd_partial
                                            current_equity_change_this_bar += net_pnl_usd_partial; total_spread_cost += spread_cost_usd_partial; total_commission_paid += commission_usd_partial; total_slippage_loss += abs(slippage_usd_partial); logging.debug(f"         Partial Close: Lot={lot_to_close:.2f}, PnL(Net USD)={net_pnl_usd_partial:.2f}")
                                            trade_log_entry_partial = {"period": label, "side": order_side, "entry_idx": order.get("entry_idx"), "entry_time": order.get("entry_time"), "entry_price": entry_price, "close_time": close_timestamp_partial, "exit_price": exit_price_ptp, "exit_reason": close_reason_partial, "lot": lot_to_close, "original_sl_price": order.get("original_sl_price", np.nan), "final_sl_price": current_sl_price_in_order, "tp_price": current_tp_price_in_order, "pnl_points_gross": pnl_points_partial, "pnl_points_net_spread": pnl_points_net_spread_partial, "pnl_usd_gross": raw_pnl_usd_partial, "commission_usd": commission_usd_partial, "spread_cost_usd": spread_cost_usd_partial, "slippage_usd": slippage_usd_partial, "pnl_usd_net": net_pnl_usd_partial, "equity_before": equity_at_start_of_bar, "equity_after": equity_at_start_of_bar + net_pnl_usd_partial, "M15_Trend_Zone": order.get("m15_trend_zone", "N/A"), "Signal_Score": order.get("signal_score", np.nan), "Trade_Reason": order.get("trade_reason", "N/A"), "Session": order.get("session", "N/A"), "BE_Triggered_Time": order.get("be_triggered_time", pd.NaT), "Pattern_Label_Entry": order.get("pattern_label_entry", "N/A"), "Is_Reentry": order.get("is_reentry", False), "Is_Forced_Entry": order.get("is_forced_entry", False), "Meta_Proba_TP": order.get("meta_proba_tp", np.nan), "Meta2_Proba_TP": order.get("meta2_proba_tp", np.nan), "is_partial_tp": True, "partial_tp_level": level_idx + 1, "atr_at_entry": atr_at_entry_log, "equity_before_open": equity_before_open_log, "entry_gain_z": entry_gain_z_log, "entry_macd_smooth": entry_macd_smooth_log, "entry_candle_ratio": entry_candle_ratio_log, "entry_adx": entry_adx_log, "entry_volatility_index": entry_volatility_index_log, "trade_tag": order_trade_tag, "risk_mode_at_entry": risk_mode_at_entry_log, "active_model_at_entry": active_model_at_entry, "model_confidence_at_entry": model_confidence_at_entry}
                                            trade_log.append(trade_log_entry_partial); trade_history_list.append(close_reason_partial)
                                            order["lot"] = round(current_lot - lot_to_close, 2); current_lot = order["lot"]; partial_tp_processed_levels.add(level_idx); order["partial_tp_processed_levels"] = partial_tp_processed_levels; logging.debug(f"         Remaining Lot after PTP{level_idx+1}: {current_lot:.2f}")
                                            if len(partial_tp_processed_levels) == 1:
                                                if partial_tp_move_sl_to_entry:
                                                    if not math.isclose(current_sl_price_in_order, entry_price, rel_tol=1e-9, abs_tol=1e-9): logging.info(f"      Moving SL to Entry ({entry_price:.5f}) after PTP{level_idx+1} for order {order_entry_time}."); order["sl_price"] = entry_price; current_sl_price_in_order = entry_price
                                                    else: logging.debug(f"      SL already at entry after PTP{level_idx+1}.")
                                                else: logging.debug(f"      Not moving SL to entry after PTP{level_idx+1} (config disabled).")
                                                if order_side == "BUY": order["peak_since_tp1"] = current_high
                                                elif order_side == "SELL": order["trough_since_tp1"] = current_low
                                                logging.debug(f"         Initialized Peak/Trough tracking after PTP1.")
                                            if order["lot"] < MIN_LOT_SIZE: logging.info(f"      Closing remaining tiny lot ({order['lot']:.2f}) after Partial TP {level_idx+1}."); order_closed_this_bar = True; close_reason = f"Full Close on Partial TP {level_idx+1}"; exit_price = exit_price_ptp; close_timestamp = now; break
                                        else: logging.debug(f"      Lot to close ({lot_to_close:.2f}) for PTP{level_idx+1} is below MIN_LOT_SIZE. Marking level as processed."); partial_tp_processed_levels.add(level_idx); order["partial_tp_processed_levels"] = partial_tp_processed_levels
                                    break
                        else: logging.warning(f"   (Warning) Cannot calculate Partial TP for order {order_entry_time}: Invalid SL delta price ({sl_delta_price_ptp}).")
                    current_atr_num_early_exit = pd.to_numeric(current_atr, errors='coerce')
                    if not order_closed_this_bar and order.get("partial_tp_processed_levels") and pd.notna(current_atr_num_early_exit) and current_atr_num_early_exit > 1e-9:
                        reversal_threshold_atr = 1.5; early_exit_triggered = False
                        if order_side == "BUY":
                            peak_since_tp1 = order.get("peak_since_tp1")
                            if pd.notna(peak_since_tp1): order["peak_since_tp1"] = max(peak_since_tp1, current_high); reversal_distance = order["peak_since_tp1"] - current_low; reversal_threshold_price = reversal_threshold_atr * current_atr_num_early_exit;
                            if reversal_distance >= reversal_threshold_price: early_exit_triggered = True; close_reason = f"EarlyExit_Reversal_{reversal_threshold_atr}ATR"; exit_price = current_close
                        elif order_side == "SELL":
                            trough_since_tp1 = order.get("trough_since_tp1")
                            if pd.notna(trough_since_tp1): order["trough_since_tp1"] = min(trough_since_tp1, current_low); reversal_distance = current_high - order["trough_since_tp1"]; reversal_threshold_price = reversal_threshold_atr * current_atr_num_early_exit;
                            if reversal_distance >= reversal_threshold_price: early_exit_triggered = True; close_reason = f"EarlyExit_Reversal_{reversal_threshold_atr}ATR"; exit_price = current_close
                        if early_exit_triggered: logging.info(f"      Early Exit triggered for order {order_entry_time} at {now}. Reason: {close_reason}"); order_closed_this_bar = True; close_timestamp = now

                    if not order_closed_this_bar:
                        logging.debug(f"         Order {order_entry_time}: Checking Main Exit Conditions (BE-SL -> SL -> TP -> MaxBars)...")
                        order_closed_this_bar, exit_price, close_reason, close_timestamp = check_main_exit_conditions(order, row, current_bar_index, now)
                        exit_price_text_log = f"{exit_price:.5f}" if pd.notna(exit_price) else "NaN" # For logging if error occurs below
                        if order_closed_this_bar: logging.debug(f"         Order {order_entry_time}: Main Exit V2 Condition Met. Closed? {order_closed_this_bar}, Reason: {close_reason}, Exit Price: {exit_price_text_log}")

                    if order_closed_this_bar:
                        order_closed_this_bar_flag = True
                        # <<< [Patch B - Unified] Applied to logging in run_backtest_simulation_v34 (Order Closing) >>>
                        exit_price_str = f"{exit_price:.5f}" if pd.notna(exit_price) else "NaN"
                        logging.info(f"      Order Closing: Time={close_timestamp}, Final Reason={close_reason}, ExitPrice={exit_price_str}, EntryTime={order_entry_time}")
                        # <<< End of [Patch B - Unified] >>>
                        if pd.isna(exit_price) or (isinstance(exit_price, float) and np.isinf(exit_price)): logging.error(f"      (Error) Order Closed Error: Order {order_entry_time} closed with reason '{close_reason}' but exit_price is invalid ({exit_price})! Setting PnL to 0."); net_pnl_usd = 0.0; pnl_points = 0.0; pnl_points_net_spread = 0.0; raw_pnl_usd = 0.0; commission_usd = 0.0; spread_cost_usd = 0.0; slippage_usd = 0.0
                        else:
                            lot_size = order.get("lot", 0.0)
                            if lot_size >= MIN_LOT_SIZE:
                                if close_reason == "BE-SL": pnl_exit_price = entry_price; slippage_usd = 0.0; logging.info(f"         [Patch BE-SL PnL] BE-SL for {order_entry_time}. PnL calc based on entry_price={entry_price:.5f}, slippage=0.")
                                else: pnl_exit_price = exit_price; slippage_points = random.uniform(MIN_SLIPPAGE_POINTS, MAX_SLIPPAGE_POINTS); slippage_usd = slippage_points * (lot_size / MIN_LOT_SIZE) * POINT_VALUE; total_slippage_loss += abs(slippage_usd); logging.debug(f"         [Patch PnL] Order {order_entry_time}: Slippage Points={slippage_points:.2f}, Slippage USD={slippage_usd:.2f}")
                                pnl_points = (pnl_exit_price - entry_price) * 10.0 if order_side == "BUY" else (entry_price - pnl_exit_price) * 10.0; pnl_points_net_spread = pnl_points - SPREAD_POINTS; spread_cost_usd = SPREAD_POINTS * (lot_size / MIN_LOT_SIZE) * POINT_VALUE; total_spread_cost += spread_cost_usd; raw_pnl_usd = pnl_points_net_spread * (lot_size / MIN_LOT_SIZE) * POINT_VALUE; commission_usd = (lot_size / MIN_LOT_SIZE) * COMMISSION_PER_001_LOT; total_commission_paid += commission_usd; net_pnl_usd = raw_pnl_usd - commission_usd + slippage_usd
                                logging.info(f"         [Patch PnL Final] Closed Lot={lot_size:.2f}, PnL(Net USD)={net_pnl_usd:.2f} (Raw PNL={raw_pnl_usd:.2f}, Comm={commission_usd:.2f}, SpreadCost={spread_cost_usd:.2f}, Slip={slippage_usd:.2f})")
                            else:
                                net_pnl_usd = 0.0; pnl_points = 0.0; pnl_points_net_spread = 0.0; raw_pnl_usd = 0.0; commission_usd = 0.0; spread_cost_usd = 0.0; slippage_usd = 0.0
                                if lot_size > 0: close_reason += "_TinyLot"; logging.debug(f"         Closed tiny remaining lot ({lot_size:.2f}) with PnL=0.")
                        current_equity_change_this_bar += net_pnl_usd
                        if not close_reason.startswith("Partial TP"):
                            trade_log_entry_base = {"period": label, "side": order_side, "entry_idx": order.get("entry_idx"), "entry_time": order.get("entry_time"), "entry_price": entry_price, "close_time": close_timestamp, "exit_price": exit_price, "exit_reason": close_reason, "lot": lot_size, "original_sl_price": order.get("original_sl_price", np.nan), "final_sl_price": order.get("sl_price"), "tp_price": order.get("tp_price", np.nan), "pnl_points_gross": pnl_points, "pnl_points_net_spread": pnl_points_net_spread, "pnl_usd_gross": raw_pnl_usd, "commission_usd": commission_usd, "spread_cost_usd": spread_cost_usd, "slippage_usd": slippage_usd, "pnl_usd_net": net_pnl_usd, "equity_before": equity_at_start_of_bar, "equity_after": equity_at_start_of_bar + current_equity_change_this_bar, "M15_Trend_Zone": order.get("m15_trend_zone", "N/A"), "Signal_Score": order.get("signal_score", np.nan), "Trade_Reason": order.get("trade_reason", "N/A"), "Session": order.get("session", "N/A"), "BE_Triggered_Time": order.get("be_triggered_time", pd.NaT), "Pattern_Label_Entry": order.get("pattern_label_entry", "N/A"), "Is_Reentry": order.get("is_reentry", False), "Is_Forced_Entry": order.get("is_forced_entry", False), "Meta_Proba_TP": order.get("meta_proba_tp", np.nan), "Meta2_Proba_TP": order.get("meta2_proba_tp", np.nan), "is_partial_tp": False, "partial_tp_level": len(order.get("partial_tp_processed_levels", set())), "atr_at_entry": atr_at_entry_log, "equity_before_open": equity_before_open_log, "entry_gain_z": entry_gain_z_log, "entry_macd_smooth": entry_macd_smooth_log, "entry_candle_ratio": entry_candle_ratio_log, "entry_adx": entry_adx_log, "entry_volatility_index": entry_volatility_index_log, "trade_tag": order_trade_tag, "risk_mode_at_entry": risk_mode_at_entry_log, "active_model_at_entry": active_model_at_entry, "model_confidence_at_entry": model_confidence_at_entry}
                            trade_log.append(trade_log_entry_base); trade_history_list.append(close_reason)
                        if net_pnl_usd < 0: consecutive_losses += 1; logging.debug(f"      Loss recorded. Consecutive losses: {consecutive_losses}")
                        elif net_pnl_usd >= 0:
                            if consecutive_losses > 0: logging.debug(f"      Win/BE recorded. Resetting consecutive losses from {consecutive_losses} to 0.")
                            consecutive_losses = 0
                        last_n_full_trade_pnls.append(net_pnl_usd)
                        if len(last_n_full_trade_pnls) > SOFT_COOLDOWN_LOOKBACK: last_n_full_trade_pnls.pop(0)
                        if order.get("is_forced_entry", False):
                            if net_pnl_usd < 0:
                                forced_entry_consecutive_losses += 1; logging.debug(f"      Forced Entry Loss. Consecutive FE losses: {forced_entry_consecutive_losses}")
                                if forced_entry_consecutive_losses >= forced_entry_max_consecutive_losses: forced_entry_temporarily_disabled = True; logging.warning(f"         (Forced Entry Disabled) Temporarily disabled due to {forced_entry_consecutive_losses} consecutive losses.")
                            else:
                                if forced_entry_consecutive_losses > 0: logging.debug("      Forced Entry Win/BE. Resetting FE consecutive losses.")
                                forced_entry_consecutive_losses = 0
                                if forced_entry_temporarily_disabled: forced_entry_temporarily_disabled = False; logging.info("         (Forced Entry Enabled) Re-enabled after winning/BE forced entry trade.")
                        entry_bar_idx_log = order.get("entry_idx")
                        if entry_bar_idx_log is not None and entry_bar_idx_log in df_sim.index:
                            safe_set_datetime(df_sim, entry_bar_idx_log, f"Order_Closed_Time{label_suffix}", close_timestamp)
                            df_sim.loc[entry_bar_idx_log, f"PnL_Realized_USD{label_suffix}"] = net_pnl_usd; df_sim.loc[entry_bar_idx_log, f"Commission_USD{label_suffix}"] = commission_usd; df_sim.loc[entry_bar_idx_log, f"Spread_Cost_USD{label_suffix}"] = spread_cost_usd; df_sim.loc[entry_bar_idx_log, f"Slippage_USD{label_suffix}"] = slippage_usd; df_sim.loc[entry_bar_idx_log, f"Exit_Reason_Actual{label_suffix}"] = close_reason; df_sim.loc[entry_bar_idx_log, f"Exit_Price_Actual{label_suffix}"] = exit_price; df_sim.loc[entry_bar_idx_log, f"PnL_Points_Actual{label_suffix}"] = pnl_points_net_spread
                            safe_set_datetime(df_sim, entry_bar_idx_log, f"BE_Triggered_Time{label_suffix}", order.get("be_triggered_time", pd.NaT))
                        else: logging.warning(f"      (Warning) Could not find entry index '{entry_bar_idx_log}' in df_sim to update results for order {order_entry_time}.")
                        continue
                    else:
                        logging.debug(f"         Order {order_entry_time} remains open. Updating BE/TSL/TTP2...")
                        order, be_triggered_this_bar, tsl_updated_this_bar, be_sl_triggered_count_run, tsl_triggered_count_run = _update_open_order_state(order, current_high, current_low, current_atr, current_avg_atr, now, base_be_r_thresh, fold_sl_multiplier_base, base_tp_multiplier_config, be_sl_triggered_count_run, tsl_triggered_count_run)
                        logging.debug(f"         Appending order {order_entry_time} to next_active_orders.")
                        next_active_orders.append(order)
            # <<< [Patch C - Unified] End of try-except for order processing loop >>>
            except Exception as e_order_processing:
                logging.critical(f"   (CRITICAL) Error processing order {order.get('entry_time', 'N/A_ORDER')} for bar {idx}: {e_order_processing}", exc_info=True)
                traceback.print_exc()
                error_in_loop = True
                if 'order' in locals() and order not in next_active_orders :
                    next_active_orders.append(order)
                continue # Attempt to continue to the next order or next bar

            m15_trend = row.get("Trend_Zone", "NEUTRAL"); entry_long_signal = row.get("Entry_Long", 0) == 1; entry_short_signal = row.get("Entry_Short", 0) == 1; trade_tag = row.get("Trade_Tag", "N/A"); signal_score = pd.to_numeric(row.get("Signal_Score"), errors='coerce'); trade_reason = row.get("Trade_Reason", "NONE"); pattern_label = row.get("Pattern_Label", "Normal")
            final_m1_signal = "NONE"
            if side == "BUY" and entry_long_signal: final_m1_signal = "BUY"
            elif side == "SELL" and entry_short_signal: final_m1_signal = "SELL"
            df_sim.loc[idx, f"M15_Trend_Zone{label_suffix}"] = m15_trend; df_sim.loc[idx, f"M1_Entry_Signal{label_suffix}"] = final_m1_signal; df_sim.loc[idx, f"Signal_Score{label_suffix}"] = signal_score if pd.notna(signal_score) else np.nan; df_sim.loc[idx, f"Trade_Reason{label_suffix}"] = trade_reason if final_m1_signal != "NONE" else "NONE"; df_sim.loc[idx, f"Session{label_suffix}"] = session_tag; df_sim.loc[idx, f"Trade_Tag{label_suffix}"] = current_trade_tag
            entry_allowed, block_reason_entry = is_entry_allowed(row, session_tag, consecutive_losses); open_new_order = False; is_reentry_trade = False; is_forced_entry = False
            if entry_allowed:
                if (side == "BUY" and final_m1_signal == "BUY") or (side == "SELL" and final_m1_signal == "SELL"):
                    open_new_order = True; logging.debug(f"   Standard Entry Signal detected for {side} at {now}.")
                    if USE_REENTRY and not active_orders:
                        time_since_last_tp = now - last_tp_time.get(side, min_ts) if pd.notna(now) and last_tp_time.get(side, min_ts) != min_ts else pd.Timedelta.max; reentry_window = pd.Timedelta(minutes=REENTRY_COOLDOWN_BARS * TIMEFRAME_MINUTES_M1)
                        if pd.Timedelta(0) < time_since_last_tp <= reentry_window: is_reentry_trade = True; is_forced_entry = False; logging.info(f"   Re-Entry condition met for {side} at {now} (Time since last TP: {time_since_last_tp}).")
                        else: logging.debug(f"   Re-Entry condition NOT met for {side} at {now} (Time since last TP: {time_since_last_tp} > {reentry_window}).")
                    elif USE_REENTRY and active_orders: logging.debug(f"   Re-Entry skipped for {side} at {now} (Active orders exist).")
            else: logging.debug(f"   Standard entry blocked at {now}. Reason: {block_reason_entry}")
            if not open_new_order and ENABLE_FORCED_ENTRY and not forced_entry_temporarily_disabled:
                if bars_since_last_trade >= FORCED_ENTRY_BAR_THRESHOLD:
                    logging.debug(f"   Checking Forced Entry conditions at {now} (Bars since last trade: {bars_since_last_trade})...")
                    fe_market_cond_met = True; block_reason_fe = "N/A"
                    if FORCED_ENTRY_CHECK_MARKET_COND:
                        atr_fe = pd.to_numeric(row.get("ATR_14"), errors='coerce'); avg_atr_fe = pd.to_numeric(row.get("ATR_14_Rolling_Avg"), errors='coerce'); gain_z_fe = pd.to_numeric(row.get("Gain_Z"), errors='coerce'); pattern_fe = row.get("Pattern_Label", "Normal")
                        if pd.isna(atr_fe) or pd.isna(avg_atr_fe) or pd.isna(gain_z_fe): fe_market_cond_met = False; block_reason_fe = "FE_NAN_COND"
                        elif avg_atr_fe > 1e-9 and atr_fe > (avg_atr_fe * FORCED_ENTRY_MAX_ATR_MULT): fe_market_cond_met = False; block_reason_fe = "FE_HIGH_ATR"
                        elif abs(gain_z_fe) < local_forced_entry_min_gain_z_abs: fe_market_cond_met = False; block_reason_fe = "FE_LOW_GAINZ"
                        elif pattern_fe not in FORCED_ENTRY_ALLOWED_REGIMES: fe_market_cond_met = False; block_reason_fe = f"FE_BAD_REGIME({pattern_fe})"
                        logging.debug(f"      FE Market Conditions Met: {fe_market_cond_met} (Reason if False: {block_reason_fe})")
                    fe_signal_score_met = False
                    if pd.notna(signal_score): fe_signal_score_met = abs(signal_score) >= FORCED_ENTRY_MIN_SIGNAL_SCORE
                    logging.debug(f"      FE Signal Score Met: {fe_signal_score_met} (Score: {signal_score:.2f} vs Threshold: {FORCED_ENTRY_MIN_SIGNAL_SCORE:.2f})")
                    if fe_market_cond_met and fe_signal_score_met:
                        fe_side = "BUY" if signal_score > 0 else "SELL"
                        if fe_side == side: open_new_order = True; is_forced_entry = True; is_reentry_trade = False; logging.info(f"      (Forced Entry Triggered) Side: {side}, Bars since last: {bars_since_last_trade}, Score: {signal_score:.2f}")
                        else: logging.debug(f"      FE condition met, but signal side ({fe_side}) does not match simulation side ({side}).")
                    elif not fe_market_cond_met: logging.debug(f"      FE blocked by Market Condition: {block_reason_fe}")
                    elif not fe_signal_score_met: logging.debug(f"      FE blocked by Low Signal Score.")
            if open_new_order or order_closed_this_bar_flag or active_orders:
                if bars_since_last_trade > 0: logging.debug(f"   Resetting bars_since_last_trade from {bars_since_last_trade} due to activity.")
                bars_since_last_trade = 0
            elif not active_orders: bars_since_last_trade += 1
            if open_new_order:
                entry_type_str = 'Forced' if is_forced_entry else ('Re-Entry' if is_reentry_trade else 'Standard'); logging.info(f"   Attempting to Open New Order ({entry_type_str}) for {side} at {now}..."); can_open_order = True; block_reason = None
                current_equity_check = equity_at_start_of_bar + current_equity_change_this_bar; potential_peak_check = max(peak_equity, current_equity_check); current_dd_check = (potential_peak_check - current_equity_check) / potential_peak_check if potential_peak_check > 1e-9 else 0.0; min_equity_level = initial_capital_segment * min_equity_threshold_pct
                if current_equity_check < min_equity_level: can_open_order = False; block_reason = f"LOW_EQUITY ({current_equity_check:.2f} < {min_equity_level:.2f})"
                if can_open_order and current_dd_check > MAX_DRAWDOWN_THRESHOLD: can_open_order = False; block_reason = f"MAX_DD ({current_dd_check*100:.1f}% > {MAX_DRAWDOWN_THRESHOLD*100:.0f}%)"; orders_blocked_by_drawdown += 1
                if can_open_order:
                    concurrent_count = sum(1 for o in active_orders if o.get("side") == side)
                    if concurrent_count >= MAX_CONCURRENT_ORDERS: can_open_order = False; block_reason = f"MAX_CONCURRENT ({concurrent_count} >= {MAX_CONCURRENT_ORDERS})"
                vol_filter_thresh = 3.5
                if can_open_order and pd.notna(current_vol_index) and current_vol_index > vol_filter_thresh: can_open_order = False; block_reason = f"HIGH_VOL_INDEX ({current_vol_index:.2f} > {vol_filter_thresh})"
                atr_filter_thresh = 3.0; score_filter_thresh = 3.5
                if can_open_order and pd.notna(current_atr) and current_atr > atr_filter_thresh and pd.notna(signal_score) and abs(signal_score) < score_filter_thresh: can_open_order = False; block_reason = f"HIGH_ATR_LOW_SCORE (ATR={current_atr:.2f}, Score={signal_score:.2f})"
                if can_open_order and pd.notna(current_macd_smooth):
                    relax_macd_cond = False; strong_signal_thresh = 4.0; strong_gainz_thresh = 1.0
                    if pd.notna(signal_score):
                        if is_forced_entry:
                            if abs(signal_score) >= strong_signal_thresh and pd.notna(current_gain_z) and current_gain_z >= strong_gainz_thresh and pattern_label in ['Breakout', 'StrongTrend']: relax_macd_cond = True
                        elif not is_forced_entry and abs(signal_score) >= strong_signal_thresh: relax_macd_cond = True
                    if not relax_macd_cond:
                        if side == "BUY" and current_macd_smooth < 0: can_open_order = False; block_reason = f"NEG_MACD_BUY (MACD={current_macd_smooth:.3f})"
                        elif side == "SELL" and current_macd_smooth > 0: can_open_order = False; block_reason = f"POS_MACD_SELL (MACD={current_macd_smooth:.3f})"
                if can_open_order and len(last_n_full_trade_pnls) >= SOFT_COOLDOWN_LOSS_COUNT:
                    recent_losses_count = sum(1 for pnl in last_n_full_trade_pnls[-SOFT_COOLDOWN_LOOKBACK:] if pnl < 0)
                    if recent_losses_count >= SOFT_COOLDOWN_LOSS_COUNT: can_open_order = False; block_reason = f"SOFT_COOLDOWN_{SOFT_COOLDOWN_LOSS_COUNT}L{SOFT_COOLDOWN_LOOKBACK}T ({recent_losses_count} losses)"
                if block_reason: logging.debug(f"      Block Reason: {block_reason}")
                active_l1_model = None; active_l1_features = None; selected_model_key = "N/A"; model_confidence = np.nan; meta_proba_tp_for_log = np.nan
                if can_open_order and USE_META_CLASSIFIER and callable(model_switcher_func):
                    logging.debug("      Applying ML Filter (L1) using Model Switcher...")
                    context = {'session': session_tag, 'drift_score': fold_config.get('drift_score', 0.0), 'signal_score': signal_score if pd.notna(signal_score) else 0.0, 'pattern': pattern_label, 'cluster': row.get('cluster', 0), 'spike_score': row.get('spike_score', 0.0), 'current_time': now, 'consecutive_losses': consecutive_losses}
                    try:
                        selected_model_key, model_confidence = model_switcher_func(context, available_models); logging.debug(f"         Switcher selected model: '{selected_model_key}', Confidence: {model_confidence}"); model_info = available_models.get(selected_model_key)
                        if model_info and model_info.get('model') and model_info.get('features'): active_l1_model = model_info['model']; active_l1_features = model_info['features']
                        else:
                            logging.warning(f"         (Warning) Switcher selected '{selected_model_key}', but model/features invalid. Falling back to 'main'."); selected_model_key = 'main'; model_info = available_models.get('main')
                            if model_info and model_info.get('model') and model_info.get('features'): active_l1_model = model_info['model']; active_l1_features = model_info['features']
                            else: logging.error("         (Error) Fallback to main model failed. Skipping ML Filter."); can_open_order = False; block_reason = "ML1_MAIN_FALLBACK_FAIL"; active_l1_model = None
                        df_sim.loc[idx, f"Active_Model{label_suffix}"] = selected_model_key; df_sim.loc[idx, f"Model_Confidence{label_suffix}"] = model_confidence
                    except Exception as e_switch:
                        logging.error(f"      (Error) Model Switcher failed: {e_switch}. Falling back to main model.", exc_info=True); selected_model_key = 'main'; model_info = available_models.get('main')
                        if model_info and model_info.get('model') and model_info.get('features'): active_l1_model = model_info['model']; active_l1_features = model_info['features']
                        else: logging.error("      (Error) Fallback to main model failed after switcher error. Skipping ML Filter."); can_open_order = False; block_reason = "ML1_SWITCH_ERR_FALLBACK_FAIL"; active_l1_model = None
                        df_sim.loc[idx, f"Active_Model{label_suffix}"] = f"ErrorFallback_{selected_model_key}"; df_sim.loc[idx, f"Model_Confidence{label_suffix}"] = np.nan
                    if active_l1_model and active_l1_features:
                        missing_ml_features = [f for f in active_l1_features if f not in row.index]
                        if missing_ml_features: logging.error(f"      (Error) ML Filter ({selected_model_key}): Missing features {missing_ml_features} in row data. Skipping filter."); can_open_order = False; block_reason = f"ML1_FEAT_MISS_{selected_model_key.upper()}"
                        else:
                            try:
                                X_ml = pd.DataFrame([row[active_l1_features]]); numeric_cols_ml = X_ml.select_dtypes(include=np.number).columns
                                if X_ml[numeric_cols_ml].isin([np.inf, -np.inf]).any().any(): X_ml[numeric_cols_ml] = X_ml[numeric_cols_ml].replace([np.inf, -np.inf], 0)
                                if X_ml[numeric_cols_ml].isnull().any().any(): X_ml[numeric_cols_ml] = X_ml[numeric_cols_ml].fillna(0)
                                cat_cols_ml = X_ml.select_dtypes(exclude=np.number).columns
                                for cat_col in cat_cols_ml: X_ml[cat_col] = X_ml[cat_col].astype(str).fillna("Missing")
                                proba_tp = active_l1_model.predict_proba(X_ml)[0, 1]; meta_proba_tp_for_log = proba_tp; logging.debug(f"         ML Model '{selected_model_key}' Predicted Proba(TP): {proba_tp:.4f}")
                                ml_threshold = current_reentry_threshold_l1 if is_reentry_trade else current_meta_threshold_l1; logging.debug(f"         Applying ML Threshold: {ml_threshold:.4f} ({'Re-Entry' if is_reentry_trade else 'Standard'})")
                                if proba_tp < ml_threshold: can_open_order = False; block_reason = f"ML1_SKIP_{selected_model_key.upper()}" if not is_reentry_trade else f"ML1_SKIP_RE_{selected_model_key.upper()}"; orders_skipped_ml_l1 += 1; logging.debug(f"      Block Reason: {block_reason} (Proba {proba_tp:.4f} < {ml_threshold:.4f})")
                            except Exception as e_ml1: logging.error(f"      (Error) ML Filter ({selected_model_key}) failed during prediction: {e_ml1}", exc_info=True); can_open_order = False; block_reason = f"ML1_ERR_{selected_model_key.upper()}" if not is_reentry_trade else f"ML1_ERR_RE_{selected_model_key.upper()}"; meta_proba_tp_for_log = np.nan
                    elif USE_META_CLASSIFIER and not active_l1_model: logging.warning(f"      (Warning) ML Filter intended but no active model available for {selected_model_key}. Allowing trade.")
                elif USE_META_CLASSIFIER and not callable(model_switcher_func): logging.debug(f"      (Info) Skipping ML Filter: model_switcher_func is not callable.")
                if not can_open_order and block_reason:
                    log_ml_skip = block_reason.startswith("ML1_SKIP") or block_reason.startswith("FE_ML_SKIP")
                    if not log_ml_skip or logging.getLogger().level <= logging.DEBUG:
                        log_entry_blocked = {"timestamp": now, "reason": block_reason, "side": side, "fund_profile": fund_profile.get('mm_mode', 'N/A'), "active_model": selected_model_key, "model_confidence": model_confidence, "meta_proba_tp": meta_proba_tp_for_log, "signal_score": signal_score if pd.notna(signal_score) else np.nan, "pattern_label": row.get("Pattern_Label", "N/A"), "is_reentry_attempt": is_reentry_trade, "is_forced_entry": is_forced_entry}
                        blocked_order_log.append(log_entry_blocked)
                        if not log_ml_skip: logging.info(f"      Order Blocked. Reason: {block_reason}")
                if can_open_order:
                    logging.info(f"      >>> Opening {side} Order ({entry_type_str}) at {now} <<<"); atr_entry = current_atr_shifted
                    if pd.isna(atr_entry) or np.isinf(atr_entry) or atr_entry < 1e-9: logging.warning(f"         (Warning) Cannot calculate SL/TP at {now}: Invalid ATR_Shifted ({atr_entry}). Skipping Order.")
                    else:
                        entry_price = current_open
                        if pd.isna(entry_price) or np.isinf(entry_price): logging.error(f"         (Error) Cannot open Order at {now}: Invalid Open price ({entry_price}).")
                        else:
                            sl_price = np.nan; tp1_price = np.nan; tp2_price = np.nan
                            if use_gain_based_exit:
                                logging.debug("         Using Gain-Based Exit (Fixed Points) due to drift override."); fixed_sl_points = 100.0; fixed_tp_points = 150.0; sl_delta_price = fixed_sl_points / 10.0; tp_delta_price = fixed_tp_points / 10.0; sl_price = entry_price - sl_delta_price if side == "BUY" else entry_price + sl_delta_price; tp1_price = entry_price + tp_delta_price if side == "BUY" else entry_price - tp_delta_price; tp2_price = tp1_price
                            else:
                                logging.debug(f"         [Patch] Using ATR-Based SL/TP. Fold SL Multiplier: {fold_sl_multiplier_base:.2f}, ATR Entry: {atr_entry:.5f}"); sl_delta_price = atr_entry * fold_sl_multiplier_base; sl_price = entry_price - sl_delta_price if side == "BUY" else entry_price + sl_delta_price; tp1_delta = sl_delta_price * 1.0; tp1_price = entry_price + tp1_delta if side == "BUY" else entry_price - tp1_delta; tp2_r = dynamic_tp2_multiplier(current_atr, current_avg_atr, base=base_tp_multiplier_config); tp2_delta = sl_delta_price * tp2_r; tp2_price = entry_price + tp2_delta if side == "BUY" else entry_price - tp2_delta
                            logging.debug(f"         Calculated SL={sl_price:.5f}, TP1={tp1_price:.5f}, TP2={tp2_price:.5f} (SL Delta Price={sl_delta_price:.5f})")
                            mm_mode = fund_profile.get('mm_mode', 'balanced'); risk_pct = fund_profile.get('risk', DEFAULT_RISK_PER_TRADE); base_lot = calculate_lot_by_fund_mode(mm_mode, risk_pct, current_equity_check, atr_entry, sl_delta_price); boosted_lot = adjust_lot_tp2_boost(trade_history_list, base_lot); final_lot, risk_mode_applied = adjust_lot_recovery_mode(boosted_lot, consecutive_losses); logging.debug(f"         Calculated Lot: Base={base_lot:.2f}, Boosted={boosted_lot:.2f}, Final={final_lot:.2f} (RiskMode Applied={risk_mode_applied})")
                            if final_lot >= MIN_LOT_SIZE:
                                entry_time = now; total_ib_lot_accumulator += final_lot; current_atr_num_ttp2 = pd.to_numeric(current_atr, errors='coerce'); enable_ttp2 = pd.notna(current_atr_num_ttp2) and current_atr_num_ttp2 > 4.0
                                new_order = {"entry_idx": idx, "entry_time": entry_time, "entry_price": entry_price, "original_lot": final_lot, "lot": final_lot, "original_sl_price": sl_price, "sl_price": sl_price, "tp_price": tp2_price, "tp1_price": tp1_price, "entry_bar_count": current_bar_index, "side": side, "m15_trend_zone": m15_trend, "trade_tag": current_trade_tag, "signal_score": signal_score if pd.notna(signal_score) else np.nan, "trade_reason": trade_reason if not is_forced_entry else f"FORCED_{trade_reason}", "session": session_tag, "pattern_label_entry": pattern_label, "be_triggered": False, "be_triggered_time": pd.NaT, "is_reentry": is_reentry_trade, "is_forced_entry": is_forced_entry, "meta_proba_tp": meta_proba_tp_for_log, "meta2_proba_tp": meta2_proba_tp_for_log, "partial_tp_processed_levels": set(), "atr_at_entry": atr_entry, "equity_before_open": current_equity_check, "entry_gain_z": current_gain_z if pd.notna(current_gain_z) else np.nan, "entry_macd_smooth": current_macd_smooth if pd.notna(current_macd_smooth) else np.nan, "entry_candle_ratio": row.get("Candle_Ratio", np.nan), "entry_adx": row.get("ADX", np.nan), "entry_volatility_index": current_vol_index if pd.notna(current_vol_index) else np.nan, "peak_since_tp1": np.nan, "trough_since_tp1": np.nan, "risk_mode_at_entry": risk_mode_applied, "use_trailing_for_tp2": enable_ttp2, "trailing_start_price": tp1_price if enable_ttp2 else np.nan, "trailing_step_r": ADAPTIVE_TSL_DEFAULT_STEP_R if enable_ttp2 else np.nan, "peak_since_ttp2_activation": np.nan, "trough_since_ttp2_activation": np.nan, "active_model_at_entry": selected_model_key, "model_confidence_at_entry": model_confidence, "tsl_activated": False, "peak_since_tsl_activation": np.nan, "trough_since_tsl_activation": np.nan}
                                next_active_orders.append(new_order); logging.info(f"         +++ ORDER OPENED: Side={side}, Lot={final_lot:.2f}, Entry={entry_price:.5f}, SL={sl_price:.5f}, TP={tp2_price:.5f}")
                                df_sim.loc[idx, f"Order_Opened{label_suffix}"] = True; df_sim.loc[idx, f"Lot_Size{label_suffix}"] = final_lot; df_sim.loc[idx, f"Entry_Price_Actual{label_suffix}"] = entry_price; df_sim.loc[idx, f"SL_Price_Actual{label_suffix}"] = sl_price; df_sim.loc[idx, f"TP_Price_Actual{label_suffix}"] = tp2_price; df_sim.loc[idx, f"ATR_At_Entry{label_suffix}"] = atr_entry; df_sim.loc[idx, f"Equity_Before_Open{label_suffix}"] = current_equity_check; df_sim.loc[idx, f"Is_Reentry{label_suffix}"] = is_reentry_trade; df_sim.loc[idx, f"Forced_Entry{label_suffix}"] = is_forced_entry; df_sim.loc[idx, f"Meta_Proba_TP{label_suffix}"] = meta_proba_tp_for_log; df_sim.loc[idx, f"Meta2_Proba_TP{label_suffix}"] = meta2_proba_tp_for_log; df_sim.loc[idx, f"Entry_Gain_Z{label_suffix}"] = current_gain_z if pd.notna(current_gain_z) else np.nan; df_sim.loc[idx, f"Entry_MACD_Smooth{label_suffix}"] = current_macd_smooth if pd.notna(current_macd_smooth) else np.nan; df_sim.loc[idx, f"Entry_Candle_Ratio{label_suffix}"] = row.get("Candle_Ratio", np.nan); df_sim.loc[idx, f"Entry_ADX{label_suffix}"] = row.get("ADX", np.nan); df_sim.loc[idx, f"Entry_Volatility_Index{label_suffix}"] = current_vol_index if pd.notna(current_vol_index) else np.nan; df_sim.loc[idx, f"Active_Model{label_suffix}"] = selected_model_key; df_sim.loc[idx, f"Model_Confidence{label_suffix}"] = model_confidence
                                if is_reentry_trade: reentry_trades_opened += 1
                                if is_forced_entry: forced_entry_trades_opened += 1
                                bars_since_last_trade = 0
                            else:
                                block_reason = f"LOT_SIZE_MIN ({final_lot:.2f} < {MIN_LOT_SIZE})"; logging.info(f"      Order Blocked. Reason: {block_reason}"); blocked_order_log.append({"timestamp": now, "reason": block_reason, "side": side, "fund_profile": fund_profile.get('mm_mode', 'N/A'), "active_model": selected_model_key, "model_confidence": model_confidence, "meta_proba_tp": meta_proba_tp_for_log, "signal_score": signal_score if pd.notna(signal_score) else np.nan, "pattern_label": row.get("Pattern_Label", "N/A"), "is_reentry_attempt": is_reentry_trade, "is_forced_entry": is_forced_entry, "calculated_lot": final_lot})

            equity = equity_at_start_of_bar + current_equity_change_this_bar
            logging.debug(f"   Equity at end of bar {current_bar_index}: {equity:.2f} (Change: {current_equity_change_this_bar:.2f})")

            if equity <= 0 and not kill_switch_activated:
                logging.warning(f"[Patch] Margin Call triggered. Equity = {equity:.2f}."); kill_switch_activated = True; kill_switch_trigger_time = now; equity = 0
                if active_orders:
                    logging.warning(f"      Force closing {len(active_orders)} orders due to Margin Call at {now}.")
                    for mc_order in active_orders: trade_log_entry_mc = {"period": label, "side": mc_order.get("side"), "entry_idx": mc_order.get("entry_idx"), "entry_time": mc_order.get("entry_time"), "entry_price": mc_order.get("entry_price"), "close_time": now, "exit_price": current_close, "exit_reason": "MARGIN_CALL", "lot": mc_order.get("lot", 0.0), "pnl_usd_net": 0.0, "is_partial_tp": False, "partial_tp_level": len(mc_order.get("partial_tp_processed_levels", set())), "risk_mode_at_entry": mc_order.get("risk_mode_at_entry", "N/A"), "active_model_at_entry": mc_order.get("active_model_at_entry", "N/A")}; trade_log.append(trade_log_entry_mc)
                    active_orders.clear()
                next_active_orders.clear(); df_sim.loc[idx, f"Equity_Realistic{label_suffix}"] = 0.0; df_sim.loc[idx, f"Max_Drawdown_At_Point{label_suffix}"] = 1.0; df_sim.loc[idx, f"Active_Order_Count{label_suffix}"] = 0; equity_history[now] = 0.0
                remaining_indices = df_sim.index[df_sim.index > idx]
                if not remaining_indices.empty: logging.info(f"      Marking remaining {len(remaining_indices)} bars with 0 equity due to Margin Call."); df_sim.loc[remaining_indices, f"Equity_Realistic{label_suffix}"] = 0.0; df_sim.loc[remaining_indices, f"Max_Drawdown_At_Point{label_suffix}"] = 1.0; df_sim.loc[remaining_indices, f"Active_Order_Count{label_suffix}"] = 0
                break

            peak_equity = max(peak_equity, equity); current_dd_final = (peak_equity - equity) / peak_equity if peak_equity > 1e-9 else 0.0; max_drawdown_pct = max(max_drawdown_pct, current_dd_final); logging.debug(f"   Drawdown: Current={current_dd_final*100:.2f}%, Max={max_drawdown_pct*100:.2f}%")
            df_sim.loc[idx, f"Max_Drawdown_At_Point{label_suffix}"] = max_drawdown_pct; df_sim.loc[idx, f"Equity_Realistic{label_suffix}"] = equity; df_sim.loc[idx, f"Active_Order_Count{label_suffix}"] = len(next_active_orders); equity_history[now] = equity

            if enable_kill_switch and not kill_switch_activated:
                logging.debug(f"   Checking Kill Switch: DD={current_dd_final*100:.2f}% (Thresh={KILL_SWITCH_MAX_DD_THRESHOLD*100:.0f}%), Losses={consecutive_losses} (Thresh={kill_switch_consecutive_losses_config})")
                if current_dd_final > KILL_SWITCH_MAX_DD_THRESHOLD: logging.warning(f"[Patch] Kill Switch triggered due to drawdown."); logging.critical(f"     (CRITICAL) KILL SWITCH ACTIVATED (Max DD): {label} at {now}. Drawdown {current_dd_final*100:.2f}% > {KILL_SWITCH_MAX_DD_THRESHOLD*100:.0f}%. Stopping simulation loop."); kill_switch_activated = True; kill_switch_trigger_time = now; break
                elif consecutive_losses >= kill_switch_consecutive_losses_config: logging.warning(f"[Patch] Kill Switch triggered due to consecutive losses."); logging.critical(f"     (CRITICAL) KILL SWITCH ACTIVATED (Consecutive Losses): {label} at {now}. Losses: {consecutive_losses} >= {kill_switch_consecutive_losses_config}. Stopping simulation loop."); kill_switch_activated = True; kill_switch_trigger_time = now; break

            previous_risk_mode = current_risk_mode
            if consecutive_losses >= recovery_mode_consecutive_losses_config:
                if current_risk_mode != "recovery": logging.info("[Patch] Activating Recovery Mode due to consecutive losses.")
                current_risk_mode = "recovery"
            else:
                if current_risk_mode == "recovery": logging.info("[Patch] Deactivating Recovery Mode.")
                current_risk_mode = "normal"
            if current_risk_mode != previous_risk_mode: logging.info(f"      [{now}] Risk Mode for *next* bar set to: {current_risk_mode} (Losses: {consecutive_losses})")
            df_sim.loc[idx, f"Risk_Mode{label_suffix}"] = current_risk_mode
            active_orders = next_active_orders
            logging.debug(f"   End of Bar {idx}. Active orders for next bar: {len(active_orders)}")
            current_bar_index += 1
    # <<< [Patch C - Unified] End of try-except for main loop >>>
    except Exception as e_loop:
        # <<< [Patch C - Unified] Log critical error and set error_in_loop flag >>>
        logging.critical(f"   (CRITICAL) Error occurred inside simulation loop for {label} at index {idx if 'idx' in locals() else 'UNKNOWN_BAR_INDEX'}: {e_loop}", exc_info=True)
        traceback.print_exc() # Print full traceback for the caught error
        error_in_loop = True
        # <<< End of [Patch C - Unified] >>>

    logging.info(f"Simulation loop finished for {label}.")

    if active_orders:
        logging.info(f"  (Closing) กำลังปิด {len(active_orders)} ออเดอร์ที่ยังเปิดอยู่ ({label}, {side}) ณ สิ้นสุดช่วงเวลา...")
        end_time = df_sim.index[-1] if not df_sim.empty else pd.Timestamp.now(tz='UTC')
        end_close_price = pd.to_numeric(df_sim["Close"].iloc[-1], errors='coerce') if not df_sim.empty else np.nan
        logging.info(f"      [Patch EoP] End of Period Close: Using Close price '{end_close_price}' at {end_time}")
        if pd.isna(end_close_price) or np.isinf(end_close_price): logging.warning(f"   (Warning) ราคา Close สุดท้าย ({end_time}) เป็น NaN/Inf. ออเดอร์ที่เหลือจะถูกปิดด้วย PnL ที่อาจไม่ถูกต้อง.")
        remaining_orders_to_close = active_orders[:]; active_orders.clear()
        for order in remaining_orders_to_close:
            order_entry_time_end = order.get("entry_time", "N/A"); close_timestamp = end_time; order_side = order.get("side"); entry_price = pd.to_numeric(order.get("entry_price"), errors='coerce'); lot_size = order.get("lot", 0.0)
            net_pnl_usd = 0.0; pnl_points = 0.0; pnl_points_net_spread = 0.0; raw_pnl_usd = 0.0; commission_usd = 0.0; spread_cost_usd = 0.0; slippage_usd = 0.0
            atr_at_entry_log_end = order.get("atr_at_entry", np.nan); equity_before_open_log_end = order.get("equity_before_open", np.nan); entry_gain_z_log_end = order.get("entry_gain_z", np.nan); entry_macd_smooth_log_end = order.get("entry_macd_smooth", np.nan); entry_candle_ratio_log_end = order.get("entry_candle_ratio", np.nan); entry_adx_log_end = order.get("entry_adx", np.nan); entry_volatility_index_log_end = order.get("entry_volatility_index", np.nan)
            order_trade_tag_end = order.get("trade_tag", "N/A"); active_model_at_entry_end = order.get("active_model_at_entry", "N/A"); model_confidence_at_entry_end = order.get("model_confidence_at_entry", np.nan); risk_mode_at_entry_log_end = order.get("risk_mode_at_entry", "N/A")
            close_reason = "End of Period"; exit_price = end_close_price
            logging.debug(f"      [Patch EoP] End of Period Close for order {order_entry_time_end}: Reason='{close_reason}', ExitPrice={exit_price}")
            if pd.isna(exit_price) or (isinstance(exit_price, float) and np.isinf(exit_price)):
                logging.warning(f"   (Warning) ราคา Close/Exit สุดท้าย ({end_time}) เป็น NaN/Inf. ปิดออเดอร์ PnL=0."); close_reason += " (ExitPriceNaN/Inf)"; exit_price = entry_price if pd.notna(entry_price) else 0; net_pnl_usd = 0.0
            elif pd.notna(entry_price) and not np.isinf(entry_price) and lot_size >= MIN_LOT_SIZE:
                pnl_exit_price_eop = exit_price
                pnl_points = ((pnl_exit_price_eop - entry_price) * 10.0 if order_side == "BUY" else (entry_price - pnl_exit_price_eop) * 10.0)
                pnl_points_net_spread = pnl_points - SPREAD_POINTS; spread_cost_usd = SPREAD_POINTS * (lot_size / MIN_LOT_SIZE) * POINT_VALUE; total_spread_cost += spread_cost_usd
                raw_pnl_usd = pnl_points_net_spread * (lot_size / MIN_LOT_SIZE) * POINT_VALUE; commission_usd = (lot_size / MIN_LOT_SIZE) * COMMISSION_PER_001_LOT; total_commission_paid += commission_usd
                slippage_usd = 0.0; net_pnl_usd = raw_pnl_usd - commission_usd + slippage_usd; equity += net_pnl_usd
                logging.info(f"      [Patch EoP] End of Period Close: Order={order_entry_time_end}, Reason={close_reason}, Lot={lot_size:.2f}, PnL(Net USD)={net_pnl_usd:.2f}")
            else:
                if lot_size < MIN_LOT_SIZE: close_reason += " (TinyLot)"
                else: close_reason += " (Price/Lot Invalid)"
                net_pnl_usd = 0.0; logging.warning(f"      End of Period Close for order {order_entry_time_end}: PnL set to 0. Reason: {close_reason}")
            trade_log_entry_end = {"period": label, "side": order_side, "entry_idx": order.get("entry_idx"), "entry_time": order.get("entry_time"), "entry_price": entry_price, "close_time": close_timestamp, "exit_price": exit_price, "exit_reason": close_reason, "lot": lot_size, "original_sl_price": order.get("original_sl_price", np.nan), "final_sl_price": order.get("sl_price"), "tp_price": order.get("tp_price", np.nan), "pnl_points_gross": pnl_points, "pnl_points_net_spread": pnl_points_net_spread, "pnl_usd_gross": raw_pnl_usd, "commission_usd": commission_usd, "spread_cost_usd": spread_cost_usd, "slippage_usd": slippage_usd, "pnl_usd_net": net_pnl_usd, "equity_before": equity - net_pnl_usd, "equity_after": equity, "M15_Trend_Zone": order.get("m15_trend_zone", "N/A"), "Signal_Score": order.get("signal_score", np.nan), "Trade_Reason": order.get("trade_reason", "N/A"), "Session": order.get("session", "N/A"), "BE_Triggered_Time": order.get("be_triggered_time", pd.NaT), "Pattern_Label_Entry": order.get("pattern_label_entry", "N/A"), "Is_Reentry": order.get("is_reentry", False), "Is_Forced_Entry": order.get("is_forced_entry", False), "Meta_Proba_TP": order.get("meta_proba_tp", np.nan), "Meta2_Proba_TP": order.get("meta2_proba_tp", np.nan), "is_partial_tp": False, "partial_tp_level": len(order.get("partial_tp_processed_levels", set())), "atr_at_entry": atr_at_entry_log_end, "equity_before_open": equity_before_open_log_end, "entry_gain_z": entry_gain_z_log_end, "entry_macd_smooth": entry_macd_smooth_log_end, "entry_candle_ratio": entry_candle_ratio_log_end, "entry_adx": entry_adx_log_end, "entry_volatility_index": entry_volatility_index_log_end, "trade_tag": order_trade_tag_end, "risk_mode_at_entry": risk_mode_at_entry_log_end, "active_model_at_entry": active_model_at_entry_end, "model_confidence_at_entry": model_confidence_at_entry_end}
            trade_log.append(trade_log_entry_end); trade_history_list.append(close_reason)
            if net_pnl_usd < 0: consecutive_losses += 1
            elif net_pnl_usd >= 0: consecutive_losses = 0
            last_n_full_trade_pnls.append(net_pnl_usd)
            if len(last_n_full_trade_pnls) > SOFT_COOLDOWN_LOOKBACK: last_n_full_trade_pnls.pop(0)
            entry_bar_idx_log_end = order.get("entry_idx")
            if entry_bar_idx_log_end is not None and entry_bar_idx_log_end in df_sim.index:
                safe_set_datetime(df_sim, entry_bar_idx_log_end, f"Order_Closed_Time{label_suffix}", close_timestamp)
                df_sim.loc[entry_bar_idx_log_end, f"PnL_Realized_USD{label_suffix}"] = net_pnl_usd; df_sim.loc[entry_bar_idx_log_end, f"Commission_USD{label_suffix}"] = commission_usd; df_sim.loc[entry_bar_idx_log_end, f"Spread_Cost_USD{label_suffix}"] = spread_cost_usd; df_sim.loc[entry_bar_idx_log_end, f"Slippage_USD{label_suffix}"] = slippage_usd; df_sim.loc[entry_bar_idx_log_end, f"Exit_Reason_Actual{label_suffix}"] = close_reason; df_sim.loc[entry_bar_idx_log_end, f"Exit_Price_Actual{label_suffix}"] = exit_price; df_sim.loc[entry_bar_idx_log_end, f"PnL_Points_Actual{label_suffix}"] = pnl_points_net_spread
            else: logging.warning(f"   (Warning) Could not find entry index '{entry_bar_idx_log_end}' in df_sim to update results for order {order_entry_time_end} (End of Period).")
        if end_time not in equity_history: equity_history[end_time] = equity
        if not df_sim.empty:
            last_valid_idx = df_sim.index[-1]
            if last_valid_idx in df_sim.index: df_sim.loc[last_valid_idx, f"Equity_Realistic{label_suffix}"] = equity; df_sim.loc[last_valid_idx, f"Active_Order_Count{label_suffix}"] = 0

    trade_log_df_segment = pd.DataFrame(trade_log)
    logging.info(f"Created trade log DataFrame for {label} with {len(trade_log_df_segment)} entries.")

    equity_col = f"Equity_Realistic{label_suffix}"
    if equity_col in df_sim.columns:
        logging.debug(f"Forward filling {equity_col}...")
        df_sim[equity_col] = df_sim[equity_col].ffill().fillna(initial_capital_segment)
        if not df_sim.empty:
            last_idx = df_sim.index[-1]
            df_sim.loc[last_idx, equity_col] = equity
            if last_idx not in equity_history: equity_history[last_idx] = equity
        if equity <= 0:
            try: first_zero_idx = df_sim[df_sim[equity_col] <= 0].index[0]; df_sim.loc[first_zero_idx:, equity_col] = 0.0; logging.debug(f"Set equity to 0 from {first_zero_idx} due to margin call (final check).")
            except IndexError: pass

    dd_col = f"Max_Drawdown_At_Point{label_suffix}"
    if dd_col in df_sim.columns:
        logging.debug(f"Forward filling {dd_col}...")
        df_sim[dd_col] = df_sim[dd_col].ffill().fillna(0.0)
        if equity <= 0 and not df_sim.empty:
            last_idx = df_sim.index[-1]
            if last_idx in df_sim.index: df_sim.loc[last_idx, dd_col] = 1.0; logging.debug("Set final max drawdown to 1.0 due to margin call (final check).")

    summary_msg = (f"  (Finished) {label} ({side}) เสร็จสมบูรณ์. Trades:{len(trade_log_df_segment)}, "
                   f"ReEntries:{reentry_trades_opened}, Forced:{forced_entry_trades_opened}, "
                   f"BEs:{be_sl_triggered_count_run}, TSLs:{tsl_triggered_count_run}")
    logging.info(summary_msg)
    logging.info(f"      Equity สุดท้าย: ${equity:.2f} (จาก ${initial_capital_segment:.2f})")
    logging.info(f"      Blocks: MaxDD={orders_blocked_by_drawdown}, Cooldown={orders_blocked_by_cooldown}, LotScale={orders_lot_scaled}, ML1Skip={orders_skipped_ml_l1}(T={current_meta_threshold_l1:.2f})")
    new_blocks_count = sum(1 for b in blocked_order_log if b.get('reason') in ["HIGH_VOL_INDEX", "HIGH_ATR_LOW_SCORE", "NEG_MACD_BUY", "POS_MACD_SELL", f"SOFT_COOLDOWN_{SOFT_COOLDOWN_LOSS_COUNT}L{SOFT_COOLDOWN_LOOKBACK}T", "SPIKE_GUARD_LONDON"])
    logging.info(f"      Blocks (New v4.6/v4.8): Vol/ATR/MACD/SoftCool/Spike={new_blocks_count}")
    if enable_kill_switch and kill_switch_activated: logging.warning(f"      *** KILL SWITCH ACTIVATED during this run! ***")
    if forced_entry_temporarily_disabled: logging.warning(f"      *** Forced Entry was temporarily disabled during this run due to loss streak. ***")
    if current_risk_mode == "recovery": logging.warning(f"      *** Ended run in RECOVERY MODE (Losses: {consecutive_losses}) ***")

    # <<< [Patch C - Unified] Add error_in_loop to run_summary >>>
    if 'run_summary' not in locals(): run_summary = {} # Should have been initialized
    run_summary.update({
        "error_in_loop": error_in_loop, # Add the flag here
        "total_commission": total_commission_paid, "total_spread": total_spread_cost, "total_slippage": total_slippage_loss,
        "orders_blocked_dd": orders_blocked_by_drawdown, "orders_blocked_cooldown": orders_blocked_by_cooldown,
        "orders_scaled_lot": orders_lot_scaled, "be_sl_triggered_count": be_sl_triggered_count_run,
        "tsl_triggered_count": tsl_triggered_count_run, "orders_skipped_ml_l1": orders_skipped_ml_l1,
        "orders_skipped_ml_l2": orders_skipped_ml_l2, "reentry_trades_opened": reentry_trades_opened,
        "forced_entry_trades_opened": forced_entry_trades_opened, "meta_model_type_l1": sim_model_type_l1,
        "meta_model_type_l2": sim_model_type_l2, "threshold_l1_used": current_meta_threshold_l1,
        "threshold_l2_used": np.nan, "kill_switch_activated": kill_switch_activated,
        "forced_entry_disabled_status": forced_entry_temporarily_disabled, "orders_blocked_new_v46": new_blocks_count,
        "drift_override_active": drift_override_active, "drift_override_reason": drift_override_reason,
        "final_risk_mode": current_risk_mode, "fund_profile": fund_profile,
        "total_ib_lot_accumulator": total_ib_lot_accumulator,
    })
    # <<< End of [Patch C - Unified] >>>
    logging.debug(f"Run Summary for {label}: {run_summary}")

    logging.debug(f"   Cleaning up memory for simulation run: {label}")
    if 'iterator' in locals(): del iterator
    if 'row' in locals(): del row
    if 'active_orders' in locals(): del active_orders
    if 'next_active_orders' in locals(): del next_active_orders
    if 'trade_log' in locals(): del trade_log
    if 'trade_history_list' in locals(): del trade_history_list
    if 'last_n_full_trade_pnls' in locals(): del last_n_full_trade_pnls
    gc.collect()
    logging.debug(f"   Memory cleanup complete for: {label}")

    return (df_sim, trade_log_df_segment, equity, equity_history, max_drawdown_pct, run_summary, blocked_order_log, sim_model_type_l1, sim_model_type_l2, kill_switch_activated, consecutive_losses, total_ib_lot_accumulator)

logging.info("Part 8: Backtesting Engine Functions Loaded (v4.8.8 Patch 26.5.1 Applied).")
# === END OF PART 8/12 ===


# === START OF PART 9/12 ===

# ==============================================================================
# === PART 9: Walk-Forward Orchestration & Analysis (v4.8.3 Patch 1) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, enhanced analysis robustness, fixed SyntaxError, added memory cleanup >>>
# <<< MODIFIED v4.8.1: Added input validation and handling for no trades in run_all_folds_with_threshold >>>
# <<< MODIFIED v4.8.3: Applied SyntaxError fix for try-except global variable checks >>>
import logging
import os
import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_ind, wasserstein_distance # For DriftObserver
from sklearn.model_selection import TimeSeriesSplit # For Walk-Forward
import gc # For memory management

# Ensure global configurations are accessible if run independently
# Define defaults if globals are not found
DEFAULT_DRIFT_WASSERSTEIN_THRESHOLD = 0.1
DEFAULT_DRIFT_TTEST_ALPHA = 0.05
DEFAULT_INITIAL_CAPITAL = 100.0
DEFAULT_IB_COMMISSION_PER_LOT = 7.0
DEFAULT_N_WALK_FORWARD_SPLITS = 5
DEFAULT_ENTRY_CONFIG_PER_FOLD = {0: {}} # Minimal default
DEFAULT_FUND_PROFILES = {"NORMAL": {"risk": 0.01, "mm_mode": "balanced"}}
DEFAULT_FUND_NAME = "NORMAL"
DEFAULT_META_MIN_PROBA_THRESH = 0.55
DEFAULT_ENABLE_PARTIAL_TP = True
DEFAULT_PARTIAL_TP_LEVELS = []
DEFAULT_PARTIAL_TP_MOVE_SL_TO_ENTRY = True
DEFAULT_ENABLE_KILL_SWITCH = True
DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD = 0.20
DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 7
DEFAULT_RECOVERY_MODE_CONSECUTIVE_LOSSES = 4
DEFAULT_min_equity_threshold_pct = 0.70
DEFAULT_DYNAMIC_GAINZ_DRIFT_THRESHOLD = 0.10
DEFAULT_DYNAMIC_GAINZ_ADJUSTMENT = 0.1

try:
    DRIFT_WASSERSTEIN_THRESHOLD
except NameError:
    DRIFT_WASSERSTEIN_THRESHOLD = DEFAULT_DRIFT_WASSERSTEIN_THRESHOLD
try:
    DRIFT_TTEST_ALPHA
except NameError:
    DRIFT_TTEST_ALPHA = DEFAULT_DRIFT_TTEST_ALPHA
try:
    INITIAL_CAPITAL
except NameError:
    INITIAL_CAPITAL = DEFAULT_INITIAL_CAPITAL
try:
    IB_COMMISSION_PER_LOT
except NameError:
    IB_COMMISSION_PER_LOT = DEFAULT_IB_COMMISSION_PER_LOT
try:
    N_WALK_FORWARD_SPLITS
except NameError:
    N_WALK_FORWARD_SPLITS = DEFAULT_N_WALK_FORWARD_SPLITS
try:
    ENTRY_CONFIG_PER_FOLD
except NameError:
    ENTRY_CONFIG_PER_FOLD = DEFAULT_ENTRY_CONFIG_PER_FOLD
try:
    FUND_PROFILES
except NameError:
    FUND_PROFILES = DEFAULT_FUND_PROFILES
try:
    DEFAULT_FUND_NAME
except NameError:
    DEFAULT_FUND_NAME = DEFAULT_FUND_NAME
try:
    META_MIN_PROBA_THRESH
except NameError:
    META_MIN_PROBA_THRESH = DEFAULT_META_MIN_PROBA_THRESH
try:
    ENABLE_PARTIAL_TP
except NameError:
    ENABLE_PARTIAL_TP = DEFAULT_ENABLE_PARTIAL_TP
try:
    PARTIAL_TP_LEVELS
except NameError:
    PARTIAL_TP_LEVELS = DEFAULT_PARTIAL_TP_LEVELS
try:
    PARTIAL_TP_MOVE_SL_TO_ENTRY
except NameError:
    PARTIAL_TP_MOVE_SL_TO_ENTRY = DEFAULT_PARTIAL_TP_MOVE_SL_TO_ENTRY
try:
    ENABLE_KILL_SWITCH
except NameError:
    ENABLE_KILL_SWITCH = DEFAULT_ENABLE_KILL_SWITCH
try:
    KILL_SWITCH_MAX_DD_THRESHOLD
except NameError:
    KILL_SWITCH_MAX_DD_THRESHOLD = DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD
try:
    KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD
except NameError:
    KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD
try:
    RECOVERY_MODE_CONSECUTIVE_LOSSES
except NameError:
    RECOVERY_MODE_CONSECUTIVE_LOSSES = DEFAULT_RECOVERY_MODE_CONSECUTIVE_LOSSES
try:
    min_equity_threshold_pct
except NameError:
    min_equity_threshold_pct = DEFAULT_min_equity_threshold_pct
try:
    DYNAMIC_GAINZ_DRIFT_THRESHOLD
except NameError:
    DYNAMIC_GAINZ_DRIFT_THRESHOLD = DEFAULT_DYNAMIC_GAINZ_DRIFT_THRESHOLD
try:
    DYNAMIC_GAINZ_ADJUSTMENT
except NameError:
    DYNAMIC_GAINZ_ADJUSTMENT = DEFAULT_DYNAMIC_GAINZ_ADJUSTMENT

# --- Drift Observer Class ---
class DriftObserver:
    """
    Observes and analyzes feature drift between training and testing folds
    using Wasserstein distance and T-tests.
    """
    def __init__(self, features_to_observe):
        """
        Initializes the DriftObserver.

        Args:
            features_to_observe (list): A list of feature names (strings) to monitor for drift.
        """
        if not isinstance(features_to_observe, list) or not all(isinstance(f, str) for f in features_to_observe):
            raise ValueError("features_to_observe must be a list of strings.")
        self.features = features_to_observe
        self.results = {} # Dictionary to store results per fold {fold_num: {feature: {metric: value}}}
        logging.info(f"   (DriftObserver) Initialized with {len(self.features)} features to observe.")

    def analyze_fold(self, train_df_pd, test_df_pd, fold_num):
        """
        Analyzes feature drift between the training and testing data for a specific fold.

        Args:
            train_df_pd (pd.DataFrame): Training data for the fold.
            test_df_pd (pd.DataFrame): Testing data for the fold.
            fold_num (int): The index of the current fold (starting from 0).
        """
        logging.info(f"    (DriftObserver) Analyzing Drift for Fold {fold_num + 1} (M1 Features)...")
        if not isinstance(train_df_pd, pd.DataFrame) or not isinstance(test_df_pd, pd.DataFrame):
            logging.warning(f"      (Warning) Skipping Drift analysis Fold {fold_num + 1}: Input is not a DataFrame.")
            return
        if not isinstance(fold_num, int) or fold_num < 0:
            logging.warning(f"      (Warning) Skipping Drift analysis: Invalid fold_num ({fold_num}).")
            return

        fold_results = {}
        self.results[fold_num] = fold_results

        if train_df_pd.empty or test_df_pd.empty:
            logging.warning(f"      (Warning) Skipping Drift analysis Fold {fold_num + 1}: Train or Test DataFrame is empty.")
            return

        common_features = list(set(train_df_pd.columns) & set(test_df_pd.columns))
        features_to_analyze = [f for f in self.features if f in common_features]

        missing_observed = [f for f in self.features if f not in features_to_analyze]
        if missing_observed:
            logging.warning(f"      (Warning) Missing observed features for drift analysis Fold {fold_num + 1}: {missing_observed}")

        if not features_to_analyze:
            logging.warning(f"      (Warning) No common observed features available to analyze Drift for Fold {fold_num + 1}.")
            return
        else:
            logging.info(f"      Analyzing {len(features_to_analyze)} common observed features for Fold {fold_num + 1}.")

        analyzed_count = 0; skipped_non_numeric = 0; skipped_insufficient_data = 0; error_count = 0; drift_alert_count = 0
        wasserstein_threshold = DRIFT_WASSERSTEIN_THRESHOLD
        ttest_alpha = DRIFT_TTEST_ALPHA
        features_for_drift_alert = ['Gain_Z', 'ATR_14', 'Candle_Speed', 'RSI']
        features_for_drift_warning = ['MACD_hist_smooth', 'Volatility_Index', 'ADX']
        drift_warning_threshold = wasserstein_threshold * 1.5

        for feature in features_to_analyze:
            feature_result = {"wasserstein": np.nan, "ttest_stat": np.nan, "ttest_p": np.nan}
            try:
                train_dtype = train_df_pd[feature].dtype
                test_dtype = test_df_pd[feature].dtype
                if not pd.api.types.is_numeric_dtype(train_dtype) or not pd.api.types.is_numeric_dtype(test_dtype):
                    logging.debug(f"         Skipping drift for non-numeric feature: '{feature}' (Types: {train_dtype}, {test_dtype})")
                    fold_results[feature] = feature_result
                    skipped_non_numeric += 1
                    continue

                train_series = pd.to_numeric(train_df_pd[feature], errors='coerce').dropna()
                test_series = pd.to_numeric(test_df_pd[feature], errors='coerce').dropna()

                min_data_points = 10
                if train_series.empty or test_series.empty or len(train_series) < min_data_points or len(test_series) < min_data_points:
                    logging.debug(f"         Skipping drift for feature '{feature}': Insufficient data after dropna (Train: {len(train_series)}, Test: {len(test_series)}).")
                    fold_results[feature] = feature_result
                    skipped_insufficient_data += 1
                    continue

                w_dist = wasserstein_distance(train_series, test_series)
                feature_result["wasserstein"] = w_dist

                if feature in features_for_drift_alert and w_dist > wasserstein_threshold:
                    logging.warning(f"          [DRIFT ALERT] Feature='{feature}', Fold={fold_num+1}, Wasserstein={w_dist:.4f} > {wasserstein_threshold:.2f}")
                    drift_alert_count += 1
                elif feature in features_for_drift_warning and w_dist > drift_warning_threshold:
                    logging.warning(f"          (Drift Warning) Feature='{feature}', Fold={fold_num+1}, Wasserstein={w_dist:.4f} > {drift_warning_threshold:.2f}")
                elif w_dist > wasserstein_threshold:
                    logging.info(f"          (Drift Info) Feature='{feature}', Fold={fold_num+1}, Wasserstein={w_dist:.4f} > {wasserstein_threshold:.2f}")

                t_stat, t_p = np.nan, np.nan
                if (train_series.nunique() > 1 and test_series.nunique() > 1 and
                    train_series.var(ddof=1) > 1e-9 and test_series.var(ddof=1) > 1e-9):
                    try:
                        t_stat, t_p = ttest_ind(train_series, test_series, equal_var=False, nan_policy='omit')
                        if pd.notna(t_p) and t_p < ttest_alpha:
                             logging.info(f"          (Drift Info) Feature='{feature}', Fold={fold_num+1}, T-test p={t_p:.4f} < {ttest_alpha:.2f} (Statistically significant mean difference)")
                    except Exception as e_ttest:
                         logging.warning(f"      (Warning) T-test failed for '{feature}' Fold {fold_num + 1}: {e_ttest}")
                else:
                    logging.debug(f"         Skipping T-test for feature '{feature}': Insufficient variance or unique values.")

                feature_result["ttest_stat"] = t_stat
                feature_result["ttest_p"] = t_p
                analyzed_count += 1
                del train_series, test_series, w_dist, t_stat, t_p
                gc.collect()

            except KeyError:
                logging.error(f"      (Error) Feature '{feature}' missing during analysis Fold {fold_num + 1}.")
                fold_results[feature] = feature_result; error_count += 1; continue
            except ValueError as ve:
                logging.warning(f"      (Warning) ValueError during drift calc '{feature}' Fold {fold_num + 1}: {ve}")
                error_count += 1
            except Exception as e:
                logging.error(f"      (Error) Drift calc failed for '{feature}' Fold {fold_num + 1}: {e}", exc_info=True)
                error_count += 1
            finally:
                 fold_results[feature] = feature_result

        logging.info(f"    (DriftObserver) Fold {fold_num + 1} Summary: Analyzed={analyzed_count}, Skipped(NN/Insuf)={skipped_non_numeric}/{skipped_insufficient_data}, Errors={error_count}, Drift Alerts={drift_alert_count}")
        del common_features, features_to_analyze, missing_observed
        gc.collect()

    def get_fold_drift_summary(self, fold_num):
        """
        Calculates the mean Wasserstein distance for a given fold based on analyzed features.

        Args:
            fold_num (int): The index of the fold.

        Returns:
            float: The mean Wasserstein distance for the fold, or np.nan if no valid results.
        """
        if fold_num not in self.results: return np.nan
        fold_data = self.results[fold_num]
        if not fold_data or not isinstance(fold_data, dict): return np.nan

        w_dists = [res["wasserstein"] for res in fold_data.values() if isinstance(res, dict) and pd.notna(res.get("wasserstein"))]
        mean_w_dist = np.mean(w_dists) if w_dists else np.nan
        logging.debug(f"    (DriftObserver) Fold {fold_num + 1} Mean Wasserstein: {mean_w_dist:.4f}")
        return mean_w_dist

    def summarize_and_save(self, output_dir, wasserstein_threshold=None, ttest_alpha=None):
        """
        Summarizes drift results across all analyzed folds and saves a summary CSV report.

        Args:
            output_dir (str): The directory to save the summary report.
            wasserstein_threshold (float, optional): Threshold for counting drifting features.
                                                     Defaults to global DRIFT_WASSERSTEIN_THRESHOLD.
            ttest_alpha (float, optional): Alpha level for counting drifting features via T-test.
                                           Defaults to global DRIFT_TTEST_ALPHA.
        """
        if wasserstein_threshold is None: global DRIFT_WASSERSTEIN_THRESHOLD; wasserstein_threshold = DRIFT_WASSERSTEIN_THRESHOLD
        if ttest_alpha is None: global DRIFT_TTEST_ALPHA; ttest_alpha = DRIFT_TTEST_ALPHA

        if not self.results: logging.warning("(Warning) No drift results to summarize."); return
        if not output_dir or not os.path.isdir(output_dir): logging.error(f"(Error) Invalid output directory for drift summary: {output_dir}."); return

        logging.info("\n(Processing) Summarizing M1 Feature Drift analysis results...")
        logging.info("  (Info) Skipping save of raw drift scores JSON (v3.6.8 Reduce Files).")

        summary_data = []
        wasserstein_df_data = {}

        for fold_num, fold_data in sorted(self.results.items()):
            if not fold_data or not isinstance(fold_data, dict):
                logging.warning(f"  (Warning) Invalid data for Fold {fold_num + 1}. Skipping summary for this fold.")
                continue

            fold_summary = {"Fold": fold_num + 1}
            numeric_data = {
                feat: res for feat, res in fold_data.items()
                if isinstance(res, dict) and pd.notna(res.get("wasserstein"))
            }

            if not numeric_data:
                logging.warning(f"  (Warning) No valid numeric drift data for Fold {fold_num + 1}.")
                fold_summary.update({
                    "Mean_Wasserstein": np.nan, "Max_Wasserstein": np.nan,
                    "Drift_Features_Wasserstein": 0, "Drift_Features_Ttest": 0,
                    "Total_Analyzed_Numeric_Features": 0
                })
            else:
                w_dists = [res["wasserstein"] for res in numeric_data.values()]
                p_vals = [res["ttest_p"] for res in numeric_data.values() if pd.notna(res.get("ttest_p"))]

                fold_summary["Mean_Wasserstein"] = np.mean(w_dists) if w_dists else np.nan
                fold_summary["Max_Wasserstein"] = np.max(w_dists) if w_dists else np.nan
                fold_summary["Drift_Features_Wasserstein"] = sum(1 for d in w_dists if d > wasserstein_threshold)
                fold_summary["Drift_Features_Ttest"] = sum(1 for p in p_vals if p < ttest_alpha)
                fold_summary["Total_Analyzed_Numeric_Features"] = len(w_dists)
                wasserstein_df_data[f"Fold {fold_num + 1}"] = {feat: res["wasserstein"] for feat, res in numeric_data.items()}

            summary_data.append(fold_summary)

        if not summary_data:
            logging.warning("  (Warning) No fold data available for drift summary CSV.")
            return

        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(output_dir, "drift_summary_m1_v32.csv")
        try:
            summary_df.to_csv(csv_path, index=False, encoding="utf-8", float_format="%.4f")
            logging.info(f"  (Success) Saved M1 drift summary (CSV): {csv_path}")
            logging.info("--- Drift Summary per Fold ---")
            logging.info("\n" + summary_df.to_string(index=False, float_format="%.4f"))
            logging.info("-----------------------------")
        except Exception as e:
            logging.error(f"  (Error) Failed to save drift summary CSV: {e}", exc_info=True)

        logging.info("  (Info) Skipping save of drift heatmap PNG (v3.6.8 Reduce Files).")
        del summary_df, summary_data, wasserstein_df_data
        gc.collect()

    def export_fold_summary(self, output_dir, fold_num):
        """
        Exports the detailed drift metrics for a specific fold to a CSV file.

        Args:
            output_dir (str): The directory to save the fold summary CSV.
            fold_num (int): The index of the fold to export.
        """
        if fold_num not in self.results:
            logging.warning(f"  (Warning) No drift results found for Fold {fold_num + 1} to export.")
            return
        if not output_dir or not os.path.isdir(output_dir):
            logging.error(f"  (Error) Invalid output directory for exporting Fold {fold_num + 1} drift summary: {output_dir}")
            return

        fold_data = self.results.get(fold_num)
        if not fold_data or not isinstance(fold_data, dict):
            logging.warning(f"  (Warning) Invalid or empty data for Fold {fold_num + 1}. Cannot export summary.")
            return

        try:
            fold_summary_list = []
            for feature, metrics in fold_data.items():
                if isinstance(metrics, dict):
                    metrics_copy = metrics.copy()
                    metrics_copy['feature'] = feature
                    fold_summary_list.append(metrics_copy)
                else:
                    logging.warning(f"    (Warning) Invalid metrics format for feature '{feature}' in Fold {fold_num + 1}. Skipping.")

            if not fold_summary_list:
                logging.warning(f"  (Warning) No valid feature metrics found for Fold {fold_num + 1}. Cannot export summary.")
                return

            fold_summary_df = pd.DataFrame(fold_summary_list)
            cols_order = ['feature', 'wasserstein', 'ttest_stat', 'ttest_p']
            cols_to_use = [col for col in cols_order if col in fold_summary_df.columns]
            fold_summary_df = fold_summary_df[cols_to_use]

            drift_fold_path = os.path.join(output_dir, f"drift_summary_fold{fold_num+1}.csv")
            fold_summary_df.to_csv(drift_fold_path, index=False, encoding="utf-8", float_format="%.4f")
            logging.debug(f"          (Success) Exported Drift Summary for Fold {fold_num+1}: {os.path.basename(drift_fold_path)}")
            del fold_summary_list, fold_summary_df
            gc.collect()
        except Exception as e_export_fold:
            logging.error(f"  (Error) Failed to export Drift Summary for Fold {fold_num+1}: {e_export_fold}", exc_info=True)

    def save(self, filepath):
        """Saves the DriftObserver object (currently skipped)."""
        logging.info(f"   (Info) Skipping save of DriftObserver object to {filepath} (v3.6.8 Reduce Files).")

    def load(self, filepath):
        """Loads a DriftObserver object (currently skipped)."""
        logging.info(f"   (Info) Skipping load of DriftObserver object from {filepath} (v3.6.8 Reduce Files).")
        self.results = {}
        return False


# --- Performance Metrics Calculation ---
def calculate_metrics(trade_log_df, final_equity, equity_history_segment, initial_capital=None, label="", model_type_l1="N/A", model_type_l2="N/A", run_summary=None, ib_lot_accumulator=0.0):
    """
    Calculates a comprehensive set of performance metrics from trade log and equity data.

    Args:
        trade_log_df (pd.DataFrame): DataFrame containing the trade log for the segment.
        final_equity (float): The final equity value at the end of the segment.
        equity_history_segment (dict or pd.Series): Equity curve data (Timestamp -> Equity).
        initial_capital (float, optional): Starting capital for the segment. Defaults to global INITIAL_CAPITAL.
        label (str): Label for the metrics keys (e.g., "Fold 1 Buy (NORMAL)"). Defaults to "".
        model_type_l1 (str): Identifier for the L1 model used. Defaults to "N/A".
        model_type_l2 (str): Identifier for the L2 model used. Defaults to "N/A".
        run_summary (dict, optional): Summary dictionary from the simulation run containing costs etc.
                                      Defaults to None.
        ib_lot_accumulator (float): Total lots traded in this segment for IB calculation. Defaults to 0.0.

    Returns:
        dict: A dictionary containing calculated performance metrics.
    """
    if initial_capital is None:
        global INITIAL_CAPITAL
        initial_capital = INITIAL_CAPITAL

    metrics = {}
    label = label.strip()
    logging.info(f"  (Metrics) Calculating metrics for: '{label}'...")
    metrics[f"{label} Initial Capital (USD)"] = initial_capital
    metrics[f"{label} ML Model Used (L1)"] = model_type_l1 if model_type_l1 else "N/A"
    metrics[f"{label} ML Model Used (L2)"] = model_type_l2 if model_type_l2 else "N/A"

    fund_profile_info = run_summary.get('fund_profile', {}) if run_summary else {}
    metrics[f"{label} Fund MM Mode"] = fund_profile_info.get('mm_mode', 'N/A')
    metrics[f"{label} Fund Risk Setting"] = fund_profile_info.get('risk', np.nan)

    if run_summary and isinstance(run_summary, dict):
        metrics[f"{label} Final Risk Mode"] = run_summary.get("final_risk_mode", "N/A")

    default_trade_metrics = {
        f"{label} Total Trades (Full)": 0, f"{label} Total Net Profit (USD)": 0.0,
        f"{label} Gross Profit (USD)": 0.0, f"{label} Gross Loss (USD)": 0.0,
        f"{label} Profit Factor": 0.0, f"{label} Average Trade (Full) (USD)": 0.0,
        f"{label} Max Trade Win (Full) (USD)": 0.0, f"{label} Max Trade Loss (Full) (USD)": 0.0,
        f"{label} Total Wins (Full)": 0, f"{label} Total Losses (Full)": 0,
        f"{label} Win Rate (Full) (%)": 0.0, f"{label} Average Win (Full) (USD)": 0.0,
        f"{label} Average Loss (Full) (USD)": 0.0, f"{label} Payoff Ratio (Full)": 0.0,
        f"{label} BE-SL Exits (Full)": 0, f"{label} Expectancy (Full) (USD)": 0.0,
        f"{label} TP Rate (Executed Full Trades) (%)": 0.0,
        f"{label} Re-Entry Trades (Full)": 0, f"{label} Forced Entry Trades (Full)": 0,
        f"{label} Partial TP Exits": 0, f"{label} Partial TP Rate (%)": 0.0,
        f"{label} Entry Count": 0,
        f"{label} TP1 Hit Rate (%)": 0.0,
        f"{label} TP2 Hit Rate (%)": 0.0,
        f"{label} SL Hit Rate (%)": 0.0,
        f"{label} AUC": np.nan,
        f"{label} Total Lots Traded (IB Accumulator)": 0.0,
        f"{label} IB Commission Estimate (USD)": 0.0,
    }

    if trade_log_df is None or not isinstance(trade_log_df, pd.DataFrame) or trade_log_df.empty:
        logging.warning(f"    (Warning) No trades logged for '{label}'. Returning default metrics.")
        metrics.update(default_trade_metrics)
        metrics[f"{label} Final Equity (USD)"] = final_equity
        if initial_capital > 1e-9:
            metrics[f"{label} Return (%)"] = ((final_equity - initial_capital) / initial_capital) * 100.0
            metrics[f"{label} Absolute Profit (USD)"] = final_equity - initial_capital
        else:
            metrics[f"{label} Return (%)"] = 0.0
            metrics[f"{label} Absolute Profit (USD)"] = 0.0
        metrics[f"{label} Max Drawdown (Equity based) (%)"] = 0.0
        metrics[f"{label} Sharpe Ratio (approx)"] = 0.0
        metrics[f"{label} Sortino Ratio (approx)"] = 0.0
        metrics[f"{label} Calmar Ratio (approx)"] = 0.0
        return metrics

    logging.info(f"    Processing {len(trade_log_df)} log entries for '{label}'...")
    if "pnl_usd_net" not in trade_log_df.columns:
        logging.warning(f"    (Warning) '{label}': Missing 'pnl_usd_net'. Setting PnL to 0.")
        trade_log_df["pnl_usd_net"] = 0.0
    else:
        trade_log_df["pnl_usd_net"] = pd.to_numeric(trade_log_df["pnl_usd_net"], errors='coerce').fillna(0.0)

    if "exit_reason" not in trade_log_df.columns:
        logging.warning(f"    (Warning) '{label}': Missing 'exit_reason'. Setting to 'N/A'.")
        trade_log_df["exit_reason"] = "N/A"
    else:
        trade_log_df["exit_reason"] = trade_log_df["exit_reason"].astype(str).fillna("N/A")

    if "is_partial_tp" not in trade_log_df.columns:
        logging.info(f"    (Info) '{label}': Missing 'is_partial_tp'. Assuming no partials.")
        trade_log_df["is_partial_tp"] = False
    else:
        trade_log_df["is_partial_tp"] = pd.to_numeric(trade_log_df["is_partial_tp"], errors='coerce').fillna(0).astype(bool)

    full_trade_log_df = trade_log_df[~trade_log_df["is_partial_tp"]].copy()
    partial_tp_log_df = trade_log_df[trade_log_df["is_partial_tp"]].copy()
    num_trades = len(full_trade_log_df)
    num_partial_tp_exits = len(partial_tp_log_df)
    total_exits = len(trade_log_df)

    metrics[f"{label} Total Trades (Full)"] = num_trades
    metrics[f"{label} Partial TP Exits"] = num_partial_tp_exits
    metrics[f"{label} Partial TP Rate (%)"] = (num_partial_tp_exits / total_exits) * 100.0 if total_exits > 0 else 0.0
    total_net_profit = trade_log_df["pnl_usd_net"].sum()
    metrics[f"{label} Total Net Profit (USD)"] = total_net_profit
    metrics[f"{label} Entry Count"] = total_exits

    if num_trades > 0:
        pnl = full_trade_log_df["pnl_usd_net"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        metrics[f"{label} Gross Profit (USD)"] = wins.sum()
        metrics[f"{label} Gross Loss (USD)"] = losses.sum()
        gp = metrics[f"{label} Gross Profit (USD)"]
        gl_abs = abs(metrics[f"{label} Gross Loss (USD)"])

        if gl_abs > 1e-9: metrics[f"{label} Profit Factor"] = gp / gl_abs
        elif gp > 0: metrics[f"{label} Profit Factor"] = np.inf
        else: metrics[f"{label} Profit Factor"] = 0.0

        metrics[f"{label} Average Trade (Full) (USD)"] = pnl.mean()
        metrics[f"{label} Max Trade Win (Full) (USD)"] = wins.max() if not wins.empty else 0.0
        metrics[f"{label} Max Trade Loss (Full) (USD)"] = losses.min() if not losses.empty else 0.0
        win_count = len(wins)
        loss_count = len(losses)
        metrics[f"{label} Total Wins (Full)"] = win_count
        metrics[f"{label} Total Losses (Full)"] = loss_count
        metrics[f"{label} Win Rate (Full) (%)"] = (win_count / num_trades) * 100.0

        avg_win = wins.mean() if win_count > 0 else 0.0
        avg_loss = losses.mean() if loss_count > 0 else 0.0
        metrics[f"{label} Average Win (Full) (USD)"] = avg_win
        metrics[f"{label} Average Loss (Full) (USD)"] = avg_loss

        avg_loss_abs = abs(avg_loss)
        if avg_loss_abs > 1e-9: metrics[f"{label} Payoff Ratio (Full)"] = avg_win / avg_loss_abs
        elif avg_win > 0: metrics[f"{label} Payoff Ratio (Full)"] = np.inf
        else: metrics[f"{label} Payoff Ratio (Full)"] = 0.0

        metrics[f"{label} BE-SL Exits (Full)"] = (full_trade_log_df["exit_reason"].str.upper() == "BE-SL").sum()

        tp_hits = (full_trade_log_df["exit_reason"].str.upper() == "TP").sum()
        metrics[f"{label} TP Rate (Executed Full Trades) (%)"] = (tp_hits / num_trades) * 100.0 if num_trades > 0 else 0.0

        wr_dec = metrics[f"{label} Win Rate (Full) (%)"] / 100.0
        metrics[f"{label} Expectancy (Full) (USD)"] = (avg_win * wr_dec) + (avg_loss * (1.0 - wr_dec))

        if "Is_Reentry" in full_trade_log_df.columns:
            try: metrics[f"{label} Re-Entry Trades (Full)"] = full_trade_log_df["Is_Reentry"].astype(bool).sum()
            except Exception: metrics[f"{label} Re-Entry Trades (Full)"] = 0
        else: metrics[f"{label} Re-Entry Trades (Full)"] = 0
        if "Is_Forced_Entry" in full_trade_log_df.columns:
            try: metrics[f"{label} Forced Entry Trades (Full)"] = full_trade_log_df["Is_Forced_Entry"].astype(bool).sum()
            except Exception: metrics[f"{label} Forced Entry Trades (Full)"] = 0
        else: metrics[f"{label} Forced Entry Trades (Full)"] = 0

        tp1_hits_partial = (partial_tp_log_df['partial_tp_level'] >= 1).sum() if 'partial_tp_level' in partial_tp_log_df.columns else 0
        tp1_hit_count = tp1_hits_partial + tp_hits
        metrics[f"{label} TP1 Hit Rate (%)"] = (tp1_hit_count / total_exits) * 100.0 if total_exits > 0 else 0.0
        metrics[f"{label} TP2 Hit Rate (%)"] = (tp_hits / total_exits) * 100.0 if total_exits > 0 else 0.0
        sl_like_exits = full_trade_log_df[~full_trade_log_df['exit_reason'].str.contains("TP|BE-SL", case=False, na=False)].copy()
        sl_hit_count = len(sl_like_exits)
        metrics[f"{label} SL Hit Rate (%)"] = (sl_hit_count / total_exits) * 100.0 if total_exits > 0 else 0.0
        del pnl, wins, losses, sl_like_exits
        gc.collect()

    else:
        logging.info(f"    No fully closed trades found for '{label}'. Calculating metrics based on partials/totals.")
        metrics.update(default_trade_metrics)
        metrics[f"{label} Total Net Profit (USD)"] = total_net_profit
        metrics[f"{label} Partial TP Exits"] = num_partial_tp_exits
        metrics[f"{label} Partial TP Rate (%)"] = (num_partial_tp_exits / total_exits) * 100.0 if total_exits > 0 else 0.0
        metrics[f"{label} Entry Count"] = total_exits
        tp1_hits_partial = (partial_tp_log_df['partial_tp_level'] >= 1).sum() if 'partial_tp_level' in partial_tp_log_df.columns else 0
        metrics[f"{label} TP1 Hit Rate (%)"] = (tp1_hits_partial / total_exits) * 100.0 if total_exits > 0 else 0.0

    metrics[f"{label} Total Lots Traded (IB Accumulator)"] = ib_lot_accumulator
    global IB_COMMISSION_PER_LOT
    metrics[f"{label} IB Commission Estimate (USD)"] = ib_lot_accumulator * IB_COMMISSION_PER_LOT

    metrics[f"{label} Final Equity (USD)"] = final_equity
    if initial_capital > 1e-9:
        metrics[f"{label} Return (%)"] = ((final_equity - initial_capital) / initial_capital) * 100.0
        metrics[f"{label} Absolute Profit (USD)"] = final_equity - initial_capital
    else:
        metrics[f"{label} Return (%)"] = 0.0
        metrics[f"{label} Absolute Profit (USD)"] = 0.0
    metrics[f"{label} Max Drawdown (Equity based) (%)"] = 0.0
    metrics[f"{label} Sharpe Ratio (approx)"] = 0.0
    metrics[f"{label} Sortino Ratio (approx)"] = 0.0
    metrics[f"{label} Calmar Ratio (approx)"] = 0.0

    equity_series = None
    logging.debug(f"    Processing equity history for '{label}'...")
    if isinstance(equity_history_segment, pd.Series):
        equity_series = equity_history_segment.copy()
    elif isinstance(equity_history_segment, dict) and equity_history_segment:
        try:
            equity_series = pd.Series({pd.to_datetime(k, errors='coerce'): v for k, v in equity_history_segment.items()})
            equity_series.dropna(inplace=True)
            if not equity_series.empty:
                equity_series.sort_index(inplace=True)
                equity_series = equity_series[~equity_series.index.duplicated(keep='last')]
                logging.debug(f"    Converted equity dict to Series (Length: {len(equity_series)}).")
            else:
                logging.warning(f"    Equity history dict for '{label}' resulted in empty Series after cleaning.")
                equity_series = None
        except Exception as e_conv:
            logging.error(f"    (Error) Error converting equity history dict to Series for '{label}': {e_conv}", exc_info=True)
            equity_series = None
    elif equity_history_segment is None or not equity_history_segment:
        logging.warning(f"    (Warning) Equity history for '{label}' is None or empty.")
    else:
        logging.warning(f"    (Warning) Unsupported equity history type for '{label}': {type(equity_history_segment)}")

    if equity_series is not None and len(equity_series) > 1:
        try:
            logging.debug(f"    Calculating Drawdown and Ratios for '{label}'...")
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max.replace(0, np.nan)
            max_dd_val = drawdown.min()
            max_dd_pct = abs(max_dd_val * 100.0) if pd.notna(max_dd_val) else 0.0
            metrics[f"{label} Max Drawdown (Equity based) (%)"] = max_dd_pct
            logging.debug(f"      Max Drawdown: {max_dd_pct:.2f}%")

            if isinstance(equity_series.index, pd.DatetimeIndex):
                equity_resampled = equity_series.resample('B').last().ffill().dropna()
                if len(equity_resampled) > 1:
                    daily_ret = equity_resampled.pct_change().dropna()
                    if not daily_ret.empty and initial_capital > 1e-9:
                        start_eq = equity_series.iloc[0]; end_eq = equity_series.iloc[-1]
                        time_delta = equity_series.index[-1] - equity_series.index[0]
                        num_years = max(time_delta.days / 365.25, 1.0 / 252.0)
                        annualized_return = 0.0
                        if start_eq > 0:
                            total_ret_compound = (end_eq / start_eq) - 1.0
                            if (1.0 + total_ret_compound) > 0:
                                annualized_return = ((1.0 + total_ret_compound) ** (1.0 / num_years)) - 1.0
                            else:
                                annualized_return = -1.0
                        logging.debug(f"      Annualized Return (approx): {annualized_return*100:.2f}%")

                        ann_std = daily_ret.std(ddof=1) * math.sqrt(252)
                        logging.debug(f"      Annualized Std Dev (approx): {ann_std:.4f}")

                        sharpe_ratio = annualized_return / ann_std if ann_std is not None and pd.notna(ann_std) and ann_std > 1e-9 else (np.inf if annualized_return > 0 else 0.0)
                        if isinstance(sharpe_ratio, complex): sharpe_ratio = 0.0
                        metrics[f"{label} Sharpe Ratio (approx)"] = sharpe_ratio
                        logging.debug(f"      Sharpe Ratio (approx): {sharpe_ratio:.3f}")

                        sortino_ratio = 0.0
                        downside_ret = daily_ret[daily_ret < 0]
                        if not downside_ret.empty:
                            downside_std = downside_ret.std(ddof=1) * math.sqrt(252)
                            if downside_std is not None and pd.notna(downside_std) and downside_std > 1e-9:
                                sortino_ratio = annualized_return / downside_std
                            elif annualized_return > 0: sortino_ratio = np.inf
                        elif annualized_return >= 0: sortino_ratio = np.inf
                        else: sortino_ratio = -np.inf
                        if isinstance(sortino_ratio, complex): sortino_ratio = 0.0
                        metrics[f"{label} Sortino Ratio (approx)"] = sortino_ratio
                        logging.debug(f"      Sortino Ratio (approx): {sortino_ratio:.3f}")

                        calmar_ratio = (annualized_return * 100.0) / max_dd_pct if max_dd_pct > 1e-9 else (np.inf if annualized_return > 0 else 0.0)
                        if isinstance(calmar_ratio, complex): calmar_ratio = 0.0
                        metrics[f"{label} Calmar Ratio (approx)"] = calmar_ratio
                        logging.debug(f"      Calmar Ratio (approx): {calmar_ratio:.3f}")
                        del daily_ret, downside_ret
                        gc.collect()
                    else:
                        logging.warning(f"    (Warning) Cannot calculate ratios for '{label}': No daily returns or zero initial capital.")
                    del equity_resampled
                    gc.collect()
                else:
                    logging.warning(f"    (Warning) Not enough resampled equity data points ({len(equity_resampled)}) for ratio calculation in '{label}'.")
            else:
                logging.warning(f"    (Warning) Equity index for '{label}' is not DatetimeIndex. Cannot calculate ratio metrics.")
            del rolling_max, drawdown
            gc.collect()
        except Exception as e:
            logging.error(f"    (Error) Error calculating equity/drawdown/ratio metrics for '{label}': {e}", exc_info=True)
    else:
        logging.warning(f"    (Warning) Not enough equity data points ({len(equity_series) if equity_series is not None else 0}) to calculate Drawdown/Ratios for '{label}'.")

    del full_trade_log_df, partial_tp_log_df, equity_series
    gc.collect()

    logging.info(f"  (Metrics) Finished calculating metrics for: '{label}'.")
    return metrics

# --- Equity Curve Plotting ---
def plot_equity_curve(equity_series_data, title, initial_capital, output_dir, filename_suffix, fold_boundaries=None):
    """
    Plots the equity curve based on provided data and saves it to a file.

    Args:
        equity_series_data (dict or pd.Series): Equity data (Timestamp -> Equity).
        title (str): The title for the plot.
        initial_capital (float): The initial capital amount for plotting the baseline.
        output_dir (str): Directory to save the plot image.
        filename_suffix (str): Suffix to append to the plot filename.
        fold_boundaries (list, optional): List of Timestamps indicating the start/end of folds
                                          for plotting vertical lines. Defaults to None.
    """
    logging.info(f"\n--- (Plotting) Plotting: {title} ---")
    logging.info(f"    Filename Suffix: {filename_suffix}")
    equity_series_plot = None

    if isinstance(equity_series_data, dict):
        if equity_series_data:
            try:
                equity_series_plot = pd.Series({pd.to_datetime(k, errors='coerce'): v for k, v in equity_series_data.items()})
                equity_series_plot.dropna(inplace=True)
                if not equity_series_plot.empty:
                    equity_series_plot.sort_index(inplace=True)
                    equity_series_plot = equity_series_plot[~equity_series_plot.index.duplicated(keep='last')]
                    logging.debug(f"   Converted equity dict to Series for plotting (Length: {len(equity_series_plot)}).")
                else:
                    logging.warning(f"   (Warning) Equity data dict for '{title}' resulted in empty Series after cleaning.")
                    equity_series_plot = None
            except Exception as e_conv:
                logging.error(f"   (Error) Error converting equity dict to Series for plot '{title}': {e_conv}", exc_info=True)
                equity_series_plot = None
        else:
            logging.warning(f"   (Warning) Equity data dict is empty for '{title}'. Cannot plot curve.")
    elif isinstance(equity_series_data, pd.Series):
        if not equity_series_data.empty:
            equity_series_plot = equity_series_data.copy()
            if not isinstance(equity_series_plot.index, pd.DatetimeIndex):
                logging.warning(f"      Equity Series index is not DatetimeIndex ({equity_series_plot.index.dtype}). Converting...");
                try:
                    equity_series_plot.index = pd.to_datetime(equity_series_plot.index, errors='coerce')
                    equity_series_plot = equity_series_plot[equity_series_plot.index.notna()]
                    if equity_series_plot.empty:
                         logging.warning("      Equity Series became empty after index conversion.")
                         equity_series_plot = None
                except Exception as e_conv_idx:
                    logging.error(f"   (Error) Error converting equity index for plot '{title}': {e_conv_idx}", exc_info=True)
                    equity_series_plot = None

            if equity_series_plot is not None and isinstance(equity_series_plot.index, pd.DatetimeIndex):
                if not equity_series_plot.index.is_monotonic_increasing:
                    logging.debug("      Sorting equity series index for plot...");
                    equity_series_plot.sort_index(inplace=True)
                if equity_series_plot.index.has_duplicates:
                    rows_before_dedup = len(equity_series_plot)
                    equity_series_plot = equity_series_plot[~equity_series_plot.index.duplicated(keep='last')]
                    rows_after_dedup = len(equity_series_plot)
                    logging.debug(f"      Removed {rows_before_dedup - rows_after_dedup} duplicate indices (keeping last) for plot.")
        else:
            logging.warning(f"   (Warning) Equity data series is empty for '{title}'. Cannot plot curve.")
    else:
        logging.warning(f"   (Warning) Unsupported equity data type '{type(equity_series_data)}' for plot '{title}'.")

    if equity_series_plot is None or equity_series_plot.empty:
        logging.warning(f"   (Info) Plotting baseline equity as data was empty/invalid for '{title}'.")
        start_time_plot = pd.Timestamp.now(tz='UTC')
        if fold_boundaries and isinstance(fold_boundaries, list):
            valid_bounds = pd.to_datetime(fold_boundaries, errors='coerce').dropna().tolist()
            if valid_bounds: start_time_plot = min(valid_bounds)
        equity_series_plot = pd.Series({start_time_plot: initial_capital})

    logging.debug(f"   Generating plot for '{title}'...")
    plt.figure(figsize=(14, 8)); ax = plt.gca()
    plot_error = False
    try:
        equity_series_plot.plot(ax=ax, label="Equity", legend=True, grid=True, linewidth=1.5, color="blue");
        init_cap_color = 'red'
    except Exception as e_plot:
        logging.error(f"   (Error) Error plotting equity curve data for '{title}': {e_plot}", exc_info=True)
        init_cap_color = 'grey'
        plot_error = True

    try:
        ax.axhline(initial_capital, color=init_cap_color, linestyle=":", linewidth=1.5, label=f"Initial Capital (${initial_capital:,.2f})")
    except Exception as e_hline:
         logging.warning(f"   (Warning) Could not plot initial capital line: {e_hline}")

    plotted_labels = set()
    if fold_boundaries and isinstance(fold_boundaries, list):
        valid_bounds = pd.to_datetime(fold_boundaries, errors='coerce').dropna().tolist()
        if len(valid_bounds) >= 2:
            logging.debug(f"   Plotting {len(valid_bounds)} fold boundaries...")
            start_bound = valid_bounds[0]
            end_bound = valid_bounds[-1]
            try:
                ax.axvline(start_bound, color="green", linestyle="--", linewidth=1, label="Start Period")
                plotted_labels.add("Start Period")
            except Exception as e_bound_start: logging.warning(f"   (Warning) Cannot plot start boundary at {start_bound}: {e_bound_start}")
            for i, bound_ts in enumerate(valid_bounds[1:-1]):
                try:
                    fold_num = i + 1; label_bound = f"End Fold {fold_num}"
                    plot_label = label_bound if label_bound not in plotted_labels else "_nolegend_"
                    ax.axvline(bound_ts, color="grey", linestyle="--", linewidth=1, label=plot_label)
                    plotted_labels.add(label_bound)
                except Exception as e_bound_mid: logging.warning(f"   (Warning) Cannot plot boundary {i+1} at {bound_ts}: {e_bound_mid}")
            try:
                ax.axvline(end_bound, color="purple", linestyle="--", linewidth=1, label="End Period")
                plotted_labels.add("End Period")
            except Exception as e_bound_end: logging.warning(f"   (Warning) Cannot plot end boundary at {end_bound}: {e_bound_end}")
            try:
                ax.set_xlim(start_bound, end_bound);
            except Exception as e_xlim: logging.warning(f"   (Warning) Cannot set xlim for plot: {e_xlim}")
        else:
            logging.warning("   (Warning) Not enough valid fold boundaries provided to plot lines.")

    logging.debug("   Formatting plot...")
    font_prop = None
    try:
        current_font_family = plt.rcParams.get('font.family')
        font_family_name = current_font_family[0] if isinstance(current_font_family, list) and current_font_family else current_font_family
        if isinstance(font_family_name, str):
            font_prop = fm.FontProperties(family=font_family_name)
    except Exception as e_fontprop:
        logging.warning(f"   (Warning) Cannot get FontProperties for plot labels: {e_fontprop}")

    ax.set_title(title, fontproperties=font_prop, fontsize=14)
    ax.set_ylabel("Equity (USD)", fontproperties=font_prop, fontsize=12)
    ax.set_xlabel("Date", fontproperties=font_prop, fontsize=12)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=10)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    ax.tick_params(axis='both', which='major', labelsize=10)
    try:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))
    except Exception as e_format:
        logging.warning(f"   (Warning) Cannot set currency formatter for Y-axis: {e_format}")

    plot_filename = os.path.join(output_dir, f"equity_curve_v32_{filename_suffix}.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        logging.info(f"   (Success) Saved plot: {plot_filename}")
    except Exception as e_save:
        logging.error(f"   (Error) Failed to save plot '{plot_filename}': {e_save}", exc_info=True)
    finally:
        plt.close()
        del equity_series_plot
        gc.collect()

# --- Log Analysis Functions ---
def load_trade_log(log_file_path):
    """Loads a trade log file (CSV or GZipped CSV) into a pandas DataFrame."""
    logging.info(f"  (Log Analysis) Loading trade log: {os.path.basename(log_file_path)}")
    if not os.path.exists(log_file_path):
        logging.error(f"    (Error) Log file not found: {log_file_path}")
        return None
    try:
        date_cols = ['entry_time', 'close_time', 'BE_Triggered_Time']
        log_df = safe_load_csv_auto(log_file_path)

        if log_df is None: return None
        if log_df.empty: logging.warning("    (Warning) Loaded trade log is empty."); return log_df

        logging.info(f"    Log loaded successfully ({len(log_df)} entries).")

        logging.debug("    Processing and validating loaded trade log...")
        for col in date_cols:
            if col in log_df.columns and not pd.api.types.is_datetime64_any_dtype(log_df[col]):
                 logging.debug(f"      Converting column '{col}' to datetime...")
                 log_df[col] = pd.to_datetime(log_df[col], errors='coerce')

        essential_cols = ['pnl_usd_net', 'exit_reason', 'side', 'entry_time', 'close_time', 'lot', 'trade_tag']
        oms_cols = ['is_partial_tp', 'partial_tp_level', 'Is_Forced_Entry', 'original_sl_price', 'entry_price', 'exit_price']
        context_cols = ['Session', 'Pattern_Label_Entry', 'M15_Trend_Zone', 'active_model_at_entry', 'model_confidence_at_entry']
        required_cols = essential_cols + oms_cols + context_cols
        missing_cols = [c for c in required_cols if c not in log_df.columns]
        if missing_cols:
            logging.warning(f"    (Warning) Log file might be missing columns:")
            for col in missing_cols: logging.warning(f"      - Missing: {col}")
            if 'pnl_usd_net' not in log_df.columns:
                logging.error("    (Error) Essential column 'pnl_usd_net' is missing.")
                return None

        logging.debug("    Ensuring correct data types for analysis columns...")
        for col in ['is_partial_tp', 'Is_Forced_Entry', 'Is_Reentry']:
            if col in log_df.columns:
                log_df[col] = pd.to_numeric(log_df[col], errors='coerce').fillna(0).astype(bool)
        for col in ['pnl_usd_net', 'lot', 'original_sl_price', 'entry_price', 'exit_price', 'partial_tp_level', 'model_confidence_at_entry', 'Signal_Score', 'atr_at_entry']:
            if col in log_df.columns:
                log_df[col] = pd.to_numeric(log_df[col], errors='coerce')
        for col in ['active_model_at_entry', 'exit_reason', 'side', 'trade_tag', 'Session', 'Pattern_Label_Entry', 'M15_Trend_Zone', 'period', 'Trade_Reason']:
             if col in log_df.columns:
                 log_df[col] = log_df[col].astype(str).fillna('N/A')

        logging.debug("    Trade log processing and validation complete.")
        return log_df
    except Exception as e:
        logging.error(f"    (Error) Failed to load or parse log file: {e}", exc_info=True)
        return None

def analyze_partial_tp(log_df):
    logging.info("\n  --- Analyzing Partial Take Profit ---")
    # ... (replace print with logging) ...
    return {} # Placeholder
def analyze_kill_switch(log_df):
    logging.info("\n  --- Analyzing Kill Switch Activations ---")
    # ... (replace print with logging) ...
    return {} # Placeholder
def analyze_consecutive_losses(log_df, loss_threshold=5):
    logging.info(f"\n  --- Analyzing Consecutive Losses (Full Trades, Threshold >= {loss_threshold}) ---")
    # ... (replace print with logging) ...
    return {} # Placeholder
def analyze_forced_entry(log_df):
    logging.info("\n  --- Analyzing Forced Entry Performance ---")
    # ... (replace print with logging) ...
    return {} # Placeholder
def analyze_performance_by_exit_reason(log_df):
    logging.info("\n  --- Analyzing Performance by Exit Reason ---")
    # ... (replace print with logging) ...
    return {} # Placeholder
def analyze_performance_by_session(log_df):
    logging.info("\n  --- Analyzing Performance by Session ---")
    # ... (replace print with logging) ...
    return {} # Placeholder
def analyze_performance_by_model(log_df):
    logging.info("\n  --- Analyzing Performance by Active Model ---")
    # ... (replace print with logging) ...
    return {} # Placeholder
def plot_log_analysis_results(log_df, output_dir, suffix=""):
    logging.info("\n  --- Generating Log Analysis Plots ---")
    # ... (replace print with logging, handle plot errors) ...
    pass # Placeholder
def run_log_analysis_pipeline(log_file_path, output_dir, consecutive_loss_config=None, suffix=""):
    logging.info(f"\n--- Running Log Analysis Pipeline (Suffix: {suffix}) ---")
    # ... (replace print with logging) ...
    log_df = load_trade_log(log_file_path)
    if log_df is None:
        logging.error("  (Error) Log analysis pipeline failed: Could not load trade log.")
        return None
    # ... call other analysis functions ...
    logging.info("\n=== Full Log Analysis Report (Summary) ===")
    # ... log summary ...
    del log_df # Clean up memory
    gc.collect()
    return {} # Placeholder

# --- Dynamic Parameter Adjustment Helper ---
def adjust_gain_z_threshold_by_drift(fold_drift_results, base_thresh):
    """
    Adjusts the Gain_Z entry threshold based on observed drift in that feature for the fold.

    Args:
        fold_drift_results (dict): Drift metrics for the current fold, keyed by feature name.
        base_thresh (float): The base Gain_Z threshold from the configuration.

    Returns:
        float: The potentially adjusted Gain_Z threshold.
    """
    global DYNAMIC_GAINZ_DRIFT_THRESHOLD, DYNAMIC_GAINZ_ADJUSTMENT

    if not isinstance(fold_drift_results, dict):
        logging.warning("      (Warning) Adjust GainZ: Invalid fold_drift_results.")
        return base_thresh
    if not isinstance(base_thresh, (int, float)):
        logging.warning(f"      (Warning) Adjust GainZ: Invalid base_thresh type ({type(base_thresh)}).")
        return base_thresh

    gain_z_drift = fold_drift_results.get("Gain_Z", {}).get("wasserstein", 0.0)
    gain_z_drift = pd.to_numeric(gain_z_drift, errors='coerce')
    gain_z_drift = gain_z_drift if pd.notna(gain_z_drift) else 0.0

    if gain_z_drift > DYNAMIC_GAINZ_DRIFT_THRESHOLD:
        adj_thresh = base_thresh + DYNAMIC_GAINZ_ADJUSTMENT
        logging.info(f"      (Drift Adjust) Gain_Z Drift ({gain_z_drift:.4f} > {DYNAMIC_GAINZ_DRIFT_THRESHOLD:.2f}). Adjusting GainZ Threshold: {base_thresh:.2f} -> {adj_thresh:.2f}")
        return adj_thresh
    else:
        logging.debug(f"      (Drift Adjust) Gain_Z Drift ({gain_z_drift:.4f}) below threshold. Using base threshold: {base_thresh:.2f}")
        return base_thresh

# --- Walk-Forward Orchestration ---
def run_all_folds_with_threshold(
    fund_profile=None,
    current_l1_threshold=None,
    df_m1_final=None,
    available_models=None,
    model_switcher_func=None,
    n_walk_forward_splits=None,
    entry_config_per_fold=None,
    drift_observer=None,
    output_dir=None,
    initial_capital=None,
    pattern_label_map=None,
    default_l1_threshold=None,
    enable_partial_tp_flag=None,
    partial_tp_levels_list=None,
    partial_tp_move_sl_flag=None,
    enable_kill_switch_flag=None,
    kill_switch_dd_thresh=None,
    kill_switch_losses_config=None,
    recovery_mode_consecutive_losses_config=None,
    min_equity_threshold_pct_config=None,
):
    """
    Orchestrates the full Walk-Forward simulation across all folds for a given
    fund profile and ML threshold. Handles data splitting, drift analysis (optional),
    running simulations for BUY and SELL sides, aggregating results, and calculating metrics.

    Args:
        fund_profile (dict): Configuration for the fund being simulated.
        current_l1_threshold (float or dict): L1 ML probability threshold(s) to use.
        df_m1_final (pd.DataFrame): The final prepared M1 data with all features.
        available_models (dict, optional): Loaded ML models and features.
        model_switcher_func (callable, optional): Function to select the active model.
        n_walk_forward_splits (int, optional): Number of WFV splits. Defaults to global.
        entry_config_per_fold (dict, optional): Fold-specific configurations. Defaults to global.
        drift_observer (DriftObserver, optional): Instance for drift analysis. Defaults to None.
        output_dir (str): Directory for saving results and logs.
        initial_capital (float, optional): Initial capital for the simulation. Defaults to global.
        pattern_label_map (dict, optional): Mapping for pattern labels (unused). Defaults to None.
        default_l1_threshold (float, optional): Default L1 threshold if override is None.
        enable_partial_tp_flag (bool, optional): Enable partial TP. Defaults to global.
        partial_tp_levels_list (list, optional): Partial TP levels config. Defaults to global.
        partial_tp_move_sl_flag (bool, optional): Move SL after partial TP. Defaults to global.
        enable_kill_switch_flag (bool, optional): Enable kill switch. Defaults to global.
        kill_switch_dd_thresh (float, optional): Kill switch DD threshold. Defaults to global.
        kill_switch_losses_config (int, optional): Kill switch loss threshold. Defaults to global.
        recovery_mode_consecutive_losses_config (int, optional): Recovery mode loss threshold. Defaults to global.
        min_equity_threshold_pct_config (float, optional): Min equity % for FE. Defaults to global.

    Returns:
        tuple: A tuple containing aggregated results:
            - metrics_buy_overall (dict): Overall metrics for BUY side.
            - metrics_sell_overall (dict): Overall metrics for SELL side.
            - df_walk_forward_results_pd_final (pd.DataFrame): Combined M1 data with simulation results.
            - trade_log_wf (pd.DataFrame): Combined trade log from all folds.
            - all_equity_histories (dict): Combined equity histories for BUY and SELL.
            - all_fold_metrics (list): List of metric dictionaries for each fold.
            - first_fold_test_data (pd.DataFrame): Data from the first test fold (for SHAP).
            - model_type_l1_used_in_run (str): L1 model type identifier used.
            - model_type_l2_used_in_run (str): L2 model type identifier used.
            - total_ib_lot_accumulator_run (float): Total accumulated lots for this fund run.
            Returns None or empty structures if the run fails critically.
    """
    if n_walk_forward_splits is None: global N_WALK_FORWARD_SPLITS; n_walk_forward_splits = N_WALK_FORWARD_SPLITS
    if entry_config_per_fold is None: global ENTRY_CONFIG_PER_FOLD; entry_config_per_fold = ENTRY_CONFIG_PER_FOLD
    if initial_capital is None: global INITIAL_CAPITAL; initial_capital = INITIAL_CAPITAL
    if default_l1_threshold is None: global META_MIN_PROBA_THRESH; default_l1_threshold = META_MIN_PROBA_THRESH
    if enable_partial_tp_flag is None: global ENABLE_PARTIAL_TP; enable_partial_tp_flag = ENABLE_PARTIAL_TP
    if partial_tp_levels_list is None: global PARTIAL_TP_LEVELS; partial_tp_levels_list = PARTIAL_TP_LEVELS
    if partial_tp_move_sl_flag is None: global PARTIAL_TP_MOVE_SL_TO_ENTRY; partial_tp_move_sl_flag = PARTIAL_TP_MOVE_SL_TO_ENTRY
    if enable_kill_switch_flag is None: global ENABLE_KILL_SWITCH; enable_kill_switch_flag = ENABLE_KILL_SWITCH
    if kill_switch_dd_thresh is None: global KILL_SWITCH_MAX_DD_THRESHOLD; kill_switch_dd_thresh = KILL_SWITCH_MAX_DD_THRESHOLD
    if kill_switch_losses_config is None: global KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD; kill_switch_losses_config = KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD
    if recovery_mode_consecutive_losses_config is None: global RECOVERY_MODE_CONSECUTIVE_LOSSES; recovery_mode_consecutive_losses_config = RECOVERY_MODE_CONSECUTIVE_LOSSES
    if min_equity_threshold_pct_config is None: global min_equity_threshold_pct; min_equity_threshold_pct_config = min_equity_threshold_pct

    fund_name = fund_profile.get('name', DEFAULT_FUND_NAME) if fund_profile else DEFAULT_FUND_NAME
    is_data_prep_mode = (available_models is None and model_switcher_func is None)
    run_label = f"(Prep Data - Fund: {fund_name})" if is_data_prep_mode else f"(Final Run - Fund: {fund_name})"

    if output_dir is None:
        logging.critical(f"      [Runner {run_label}] (Error) ไม่ได้รับค่า output_dir. ไม่สามารถดำเนินการต่อได้.")
        return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0

    l1_thresh_to_use = current_l1_threshold if current_l1_threshold is not None else default_l1_threshold
    l1_thresh_display = f"{l1_thresh_to_use:.2f}" if isinstance(l1_thresh_to_use, (float, int)) else "PerFold"

    logging.info(f"      [Runner {run_label}] เริ่มต้น Full WF Sim (L1_Th={l1_thresh_display})")
    start_time_run = time.time()

    # <<< MODIFIED v4.8.1: Added input validation for df_m1_final >>>
    if df_m1_final is None or df_m1_final.empty:
        logging.error(f"      [Runner {run_label}] (Error) df_m1_final ว่างเปล่า. ไม่สามารถรัน Backtest ได้.")
        return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0

    if not is_data_prep_mode:
        if model_switcher_func is None or not callable(model_switcher_func):
            logging.critical(f"      [Runner {run_label}] (Error) model_switcher_func ไม่ถูกต้อง."); return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0
        if (available_models is None or not isinstance(available_models, dict) or not available_models):
            logging.critical(f"      [Runner {run_label}] (Error) available_models ไม่ถูกต้อง."); return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0
        main_model_info = available_models.get('main')
        if (main_model_info is None or
            main_model_info.get('model') is None or
            not main_model_info.get('features')):
            logging.critical(f"      [Runner {run_label}] (CRITICAL ERROR) Main model object หรือ features ไม่พบ/ไม่สมบูรณ์ใน available_models.")
            logging.critical("         ไม่สามารถรัน Backtest ได้เนื่องจากไม่มี Model/Features หลัก.")
            logging.critical("         (Hint: อาจเกิดจาก Trade Log ว่างเปล่า ทำให้ Auto-Train ข้ามการสร้าง Model/Features ไป)")
            return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0

    if fund_profile is None or not isinstance(fund_profile, dict) or "mm_mode" not in fund_profile or "risk" not in fund_profile:
        logging.critical(f"      [Runner {run_label}] (Error) fund_profile ไม่ถูกต้อง."); return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0

    logging.info(f"      Initializing TimeSeriesSplit with n_splits={n_walk_forward_splits}")
    tscv = TimeSeriesSplit(n_splits=n_walk_forward_splits)
    all_fold_results_df = []
    all_trade_logs = []
    all_equity_histories = {}
    all_fold_metrics = []
    all_blocked_logs = []
    previous_fold_metrics = None
    model_type_l1_used_in_run = "Switcher" if not is_data_prep_mode and USE_META_CLASSIFIER else "N/A"
    model_type_l2_used_in_run = "N/A"
    first_fold_test_data = None
    total_ib_lot_accumulator_run = 0.0

    logging.info(f"      Starting Walk-Forward loop ({n_walk_forward_splits} folds)...")
    for fold, (train_index, test_index) in enumerate(tscv.split(df_m1_final)):
        fold_start_time = time.time()
        logging.info(f"\n{'='*15} Fold {fold + 1}/{n_walk_forward_splits} ({run_label}) {'='*15}")

        current_fold_kill_switch_state = False
        current_fold_consecutive_losses = 0
        logging.debug(f"        Initializing Fold {fold+1} state: KS Active={current_fold_kill_switch_state}, Consec Losses={current_fold_consecutive_losses}")

        df_train_fold = df_m1_final.iloc[train_index]
        df_test_fold = df_m1_final.iloc[test_index].copy()
        logging.info(f"          Train period size: {len(df_train_fold)}")
        logging.info(f"          Test period: {df_test_fold.index.min()} to {df_test_fold.index.max()} (Size: {len(df_test_fold)})")

        if df_train_fold.empty and fold > 0:
            logging.warning(f"          (Warning) Skipping Fold {fold + 1}: Training data is empty."); continue
        if df_test_fold.empty:
            logging.warning(f"          (Warning) Skipping Fold {fold + 1}: Test data is empty."); continue

        critical_cols_fold = ['Open', 'High', 'Low', 'Close', 'ATR_14_Shifted']
        if df_test_fold[critical_cols_fold].isnull().any().any():
            logging.warning(f"          (Warning) Fold {fold+1}: NaNs found in critical columns before simulation. Attempting to drop...")
            initial_rows_fold = len(df_test_fold)
            df_test_fold.dropna(subset=critical_cols_fold, inplace=True)
            logging.warning(f"             Dropped {initial_rows_fold - len(df_test_fold)} rows.")
            if df_test_fold.empty:
                logging.error(f"          (Error) Test Fold {fold+1} is empty after final NaN check. Skipping Fold."); continue

        if fold == 0 and not df_test_fold.empty:
            first_fold_test_data = df_test_fold.copy()
            logging.debug("          Stored first fold test data for potential SHAP analysis.")

        fold_drift_results = {}
        fold_drift_score_mean = 0.0
        if drift_observer:
            logging.info(f"          Performing Drift Analysis for Fold {fold+1}...")
            try:
                drift_observer.analyze_fold(df_train_fold, df_test_fold, fold)
                if fold in drift_observer.results:
                    fold_drift_results = drift_observer.results.get(fold, {})
                    fold_drift_score_mean = drift_observer.get_fold_drift_summary(fold)
                    try:
                        drift_observer.export_fold_summary(output_dir, fold)
                    except AttributeError:
                        logging.warning(f"          (Warning) DriftObserver object does not have 'export_fold_summary' method.")
                    except Exception as e_export_drift:
                        logging.error(f"          (Error) Failed to export Drift Summary for Fold {fold+1}: {e_export_drift}", exc_info=True)
                logging.info(f"          Drift Analysis complete for Fold {fold+1}. Mean Wasserstein: {fold_drift_score_mean:.4f}")
            except Exception as e_drift_analyze:
                logging.error(f"          (Error) Drift analysis failed for Fold {fold+1}: {e_drift_analyze}", exc_info=True)

        logging.debug(f"          Loading configuration for Fold {fold+1}...")
        base_cfg = entry_config_per_fold.get(fold, entry_config_per_fold.get(0))
        current_cfg = base_cfg.copy()
        current_cfg['drift_score'] = fold_drift_score_mean
        logging.debug(f"          Base Fold Config: {base_cfg}")

        if fold_drift_results:
            logging.info(f"          Adjusting parameters based on drift for Fold {fold+1}...")
            current_cfg["gain_z_thresh"] = adjust_gain_z_threshold_by_drift(fold_drift_results, current_cfg["gain_z_thresh"])
            rsi_drift_score = fold_drift_results.get("RSI", {}).get("wasserstein", 0.0)
            atr_drift_score = fold_drift_results.get("ATR_14", {}).get("wasserstein", 0.0)
            if not isinstance(rsi_drift_score, (int, float, np.number)) or pd.isna(rsi_drift_score): rsi_drift_score = 0.0
            if not isinstance(atr_drift_score, (int, float, np.number)) or pd.isna(atr_drift_score): atr_drift_score = 0.0

            rsi_drift_override_threshold = 0.3
            atr_drift_override_threshold = 0.25

            current_cfg['ignore_rsi_scoring'] = False
            current_cfg['use_gain_based_exit'] = False
            if rsi_drift_score > rsi_drift_override_threshold:
                current_cfg['ignore_rsi_scoring'] = True
                logging.warning(f"          (Drift Override) RSI Drift ({rsi_drift_score:.4f} > {rsi_drift_override_threshold:.2f}) detected. Ignoring RSI scoring for Fold {fold+1}.")
            if atr_drift_score > atr_drift_override_threshold:
                current_cfg['use_gain_based_exit'] = True
                logging.warning(f"          (Drift Override) ATR Drift ({atr_drift_score:.4f} > {atr_drift_override_threshold:.2f}) detected. Using Gain-Based Exit Logic (Placeholder) for Fold {fold+1}.")
            logging.debug(f"          Adjusted Fold Config: {current_cfg}")

        start_cap_buy = initial_capital; start_cap_sell = initial_capital
        if fold > 0 and previous_fold_metrics:
            prev_buy_label = f"Fold {fold} Buy ({fund_name})"
            prev_sell_label = f"Fold {fold} Sell ({fund_name})"
            prev_eq_buy_key = f"{prev_buy_label} Final Equity (USD)"
            prev_eq_sell_key = f"{prev_sell_label} Final Equity (USD)"
            start_cap_buy = max(previous_fold_metrics.get("buy", {}).get(prev_eq_buy_key, initial_capital), 1.0)
            start_cap_sell = max(previous_fold_metrics.get("sell", {}).get(prev_eq_sell_key, initial_capital), 1.0)
            logging.info(f"          Starting Capital Fold {fold+1}: BUY=${start_cap_buy:.2f}, SELL=${start_cap_sell:.2f} (from previous fold)")
        else:
            logging.info(f"          Starting Capital Fold {fold+1}: BUY=${start_cap_buy:.2f}, SELL=${start_cap_sell:.2f} (Initial)")

        cfg_buy = current_cfg.copy()
        cfg_sell = current_cfg.copy()

        logging.info(f"   -- Running BUY Simulation Fold {fold+1} ({fund_name}) --")
        label_buy = f"Fold{fold}_BUY_{fund_name}"
        (df_buy_res, log_buy, eq_buy, hist_buy, dd_buy, costs_buy, blocked_buy, type_l1_b, type_l2_b, final_ks_state_buy, final_losses_buy, ib_lot_buy) = run_backtest_simulation_v34(
            df_test_fold, label_buy, start_cap_buy, "BUY",
            fund_profile=fund_profile, fold_config=cfg_buy,
            available_models=available_models, model_switcher_func=model_switcher_func,
            pattern_label_map=pattern_label_map, meta_min_proba_thresh_override=l1_thresh_to_use,
            current_fold_index=fold, enable_partial_tp=enable_partial_tp_flag,
            partial_tp_levels=partial_tp_levels_list, partial_tp_move_sl_to_entry=partial_tp_move_sl_flag,
            enable_kill_switch=enable_kill_switch_flag, kill_switch_max_dd_threshold=kill_switch_dd_thresh,
            kill_switch_consecutive_losses_config=kill_switch_losses_config,
            recovery_mode_consecutive_losses_config=recovery_mode_consecutive_losses_config,
            min_equity_threshold_pct=min_equity_threshold_pct_config,
            initial_kill_switch_state=current_fold_kill_switch_state,
            initial_consecutive_losses=current_fold_consecutive_losses,
        )

        logging.info(f"   -- Running SELL Simulation Fold {fold+1} ({fund_name}) --")
        label_sell = f"Fold{fold}_SELL_{fund_name}"
        (df_sell_res, log_sell, eq_sell, hist_sell, dd_sell, costs_sell, blocked_sell, type_l1_s, type_l2_s, final_ks_state_sell, final_losses_sell, ib_lot_sell) = run_backtest_simulation_v34(
            df_buy_res, label_sell, start_cap_sell, "SELL",
            fund_profile=fund_profile, fold_config=cfg_sell,
            available_models=available_models, model_switcher_func=model_switcher_func,
            pattern_label_map=pattern_label_map, meta_min_proba_thresh_override=l1_thresh_to_use,
            current_fold_index=fold, enable_partial_tp=enable_partial_tp_flag,
            partial_tp_levels=partial_tp_levels_list, partial_tp_move_sl_to_entry=partial_tp_move_sl_flag,
            enable_kill_switch=enable_kill_switch_flag, kill_switch_max_dd_threshold=kill_switch_dd_thresh,
            kill_switch_consecutive_losses_config=kill_switch_losses_config,
            recovery_mode_consecutive_losses_config=recovery_mode_consecutive_losses_config,
            min_equity_threshold_pct=min_equity_threshold_pct_config,
            initial_kill_switch_state=current_fold_kill_switch_state,
            initial_consecutive_losses=current_fold_consecutive_losses,
        )

        logging.debug(f"Storing results for Fold {fold+1}...")
        all_fold_results_df.append(df_sell_res)
        if log_buy is not None and not log_buy.empty: all_trade_logs.append(log_buy)
        if log_sell is not None and not log_sell.empty: all_trade_logs.append(log_sell)
        all_equity_histories[f"{label_buy}"] = hist_buy
        all_equity_histories[f"{label_sell}"] = hist_sell
        all_blocked_logs.extend(blocked_buy); all_blocked_logs.extend(blocked_sell)
        total_ib_lot_accumulator_run += ib_lot_buy + ib_lot_sell

        logging.info(f"   -- Calculating Metrics for Fold {fold+1} ({fund_name}) --")
        metrics_buy_fold = calculate_metrics(log_buy, eq_buy, hist_buy, start_cap_buy, f"Fold {fold+1} Buy ({fund_name})", type_l1_b, type_l2_b, costs_buy, ib_lot_buy)
        metrics_buy_fold[f"Fold {fold+1} Buy ({fund_name}) Max Drawdown (Simulated) (%)"] = dd_buy * 100.0
        metrics_buy_fold.update({f"Fold {fold+1} Buy ({fund_name}) Costs {k.replace('_', ' ').title()}": v for k, v in costs_buy.items() if k not in ["meta_model_type_l1", "meta_model_type_l2", "threshold_l1_used", "threshold_l2_used", "fund_profile", "total_ib_lot_accumulator"]})

        metrics_sell_fold = calculate_metrics(log_sell, eq_sell, hist_sell, start_cap_sell, f"Fold {fold+1} Sell ({fund_name})", type_l1_s, type_l2_s, costs_sell, ib_lot_sell)
        metrics_sell_fold[f"Fold {fold+1} Sell ({fund_name}) Max Drawdown (Simulated) (%)"] = dd_sell * 100.0
        metrics_sell_fold.update({f"Fold {fold+1} Sell ({fund_name}) Costs {k.replace('_', ' ').title()}": v for k, v in costs_sell.items() if k not in ["meta_model_type_l1", "meta_model_type_l2", "threshold_l1_used", "threshold_l2_used", "fund_profile", "total_ib_lot_accumulator"]})

        try:
            avg_score_buy = log_buy['Signal_Score'].mean() if log_buy is not None and not log_buy.empty and 'Signal_Score' in log_buy.columns else np.nan
            avg_score_sell = log_sell['Signal_Score'].mean() if log_sell is not None and not log_sell.empty and 'Signal_Score' in log_sell.columns else np.nan
            tp_rate_buy = metrics_buy_fold.get(f"Fold {fold+1} Buy ({fund_name}) TP Rate (Executed Full Trades) (%)", np.nan)
            tp_rate_sell = metrics_sell_fold.get(f"Fold {fold+1} Sell ({fund_name}) TP Rate (Executed Full Trades) (%)", np.nan)
            be_rate_buy = (log_buy['exit_reason'].str.upper() == 'BE-SL').mean() * 100.0 if log_buy is not None and not log_buy.empty and 'exit_reason' in log_buy.columns else 0.0
            be_rate_sell = (log_sell['exit_reason'].str.upper() == 'BE-SL').mean() * 100.0 if log_sell is not None and not log_sell.empty and 'exit_reason' in log_sell.columns else 0.0
            sl_rate_buy = 100.0 - (tp_rate_buy if pd.notna(tp_rate_buy) else 0.0) - be_rate_buy
            sl_rate_sell = 100.0 - (tp_rate_sell if pd.notna(tp_rate_sell) else 0.0) - be_rate_sell

            metrics_buy_fold[f"Fold {fold+1} Buy ({fund_name}) Avg Entry Signal Score"] = avg_score_buy
            metrics_buy_fold[f"Fold {fold+1} Buy ({fund_name}) SL Rate (Full Trades) (%)"] = sl_rate_buy
            metrics_sell_fold[f"Fold {fold+1} Sell ({fund_name}) Avg Entry Signal Score"] = avg_score_sell
            metrics_sell_fold[f"Fold {fold+1} Sell ({fund_name}) SL Rate (Full Trades) (%)"] = sl_rate_sell
            logging.debug(f"        Fold {fold+1} Buy Metrics: TP Rate={tp_rate_buy:.2f}%, BE Rate={be_rate_buy:.2f}%, SL Rate={sl_rate_buy:.2f}%")
            logging.debug(f"        Fold {fold+1} Sell Metrics: TP Rate={tp_rate_sell:.2f}%, BE Rate={be_rate_sell:.2f}%, SL Rate={sl_rate_sell:.2f}%")

        except Exception as e_fold_metric_log:
            logging.warning(f"    (Warning) Could not calculate/log additional fold metrics: {e_fold_metric_log}")

        current_fold_metrics = {"buy": metrics_buy_fold, "sell": metrics_sell_fold}
        all_fold_metrics.append(current_fold_metrics)
        previous_fold_metrics = current_fold_metrics

        if log_buy is not None and log_buy.empty and log_sell is not None and log_sell.empty:
            logging.warning(f"          [SUMMARY] Fold {fold+1} ({fund_name}): No trades opened. All entries blocked.")

        logging.debug(f"        Cleaning up memory after Fold {fold+1}...")
        del df_train_fold, df_test_fold, df_buy_res, df_sell_res
        del log_buy, log_sell, hist_buy, hist_sell, blocked_buy, blocked_sell
        del metrics_buy_fold, metrics_sell_fold, current_fold_metrics
        gc.collect()
        logging.debug(f"        Memory cleanup complete for Fold {fold+1}.")

        fold_duration = time.time() - fold_start_time
        logging.info(f"   (Success) Fold {fold+1} ({fund_name}) processed in: {fold_duration:.2f} seconds")

    run_duration = time.time() - start_time_run
    logging.info(f"      [Runner {run_label}] (Success) Full WF Sim completed (L1_Th={l1_thresh_display}) in {run_duration:.2f} seconds.")

    # <<< MODIFIED v4.8.1: Handle cases where no trades were logged or no metrics generated >>>
    if not all_trade_logs:
        logging.error(f"      [Runner {run_label}] (Error) No trades were logged in any fold (L1_Th={l1_thresh_display}). Cannot aggregate results.")
        return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0
    if not all_fold_metrics:
        logging.error(f"      [Runner {run_label}] (Error) No metrics were generated from any fold (L1_Th={l1_thresh_display}). Cannot aggregate results.")
        return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0

    logging.info(f"      [Runner {run_label}] (Processing) Aggregating overall results (L1_Th={l1_thresh_display})...")
    trade_log_wf = pd.DataFrame()
    if all_trade_logs:
        try:
            trade_log_wf = pd.concat(all_trade_logs, ignore_index=True)
            if "entry_time" in trade_log_wf.columns:
                trade_log_wf["entry_time"] = pd.to_datetime(trade_log_wf["entry_time"])
                trade_log_wf.sort_values(by="entry_time", inplace=True)
            else:
                logging.warning("   (Warning) Final combined trade log is missing 'entry_time' column.")
            logging.info(f"      Combined Trade Log Shape: {trade_log_wf.shape}")
            del all_trade_logs
            gc.collect()
        except Exception as e:
            logging.error(f"        (Error) Failed to concatenate trade logs (L1_Th={l1_thresh_display}): {e}.", exc_info=True)
            return None, None, pd.DataFrame(), pd.DataFrame(), {}, [], None, "N/A", "N/A", 0.0

    logging.debug("      Combining equity histories...")
    eq_buy_hist_combined = {}; eq_sell_hist_combined = {}
    for lbl, hist in all_equity_histories.items():
        if isinstance(lbl, str) and "_BUY_" in lbl: eq_buy_hist_combined.update(hist)
        elif isinstance(lbl, str) and "_SELL_" in lbl: eq_sell_hist_combined.update(hist)

    eq_buy_series_final = pd.Series(dict(sorted(eq_buy_hist_combined.items()))).sort_index()
    eq_buy_series_final = eq_buy_series_final[~eq_buy_series_final.index.duplicated(keep='last')]
    eq_sell_series_final = pd.Series(dict(sorted(eq_sell_hist_combined.items()))).sort_index()
    eq_sell_series_final = eq_sell_series_final[~eq_sell_series_final.index.duplicated(keep='last')]
    logging.debug(f"      Combined BUY Equity Series Length: {len(eq_buy_series_final)}")
    logging.debug(f"      Combined SELL Equity Series Length: {len(eq_sell_series_final)}")
    del eq_buy_hist_combined, eq_sell_hist_combined
    gc.collect()

    logging.info(f"      [Runner {run_label}] (Calculating) Calculating overall metrics (L1_Th={l1_thresh_display})...")
    metrics_buy_overall = None
    log_wf_buy = trade_log_wf[trade_log_wf["side"] == "BUY"].copy() if not trade_log_wf.empty else pd.DataFrame()
    final_eq_buy = eq_buy_series_final.iloc[-1] if not eq_buy_series_final.empty else initial_capital
    total_ib_lot_buy_run = sum(m["buy"].get(f"Fold {i+1} Buy ({fund_name}) Costs Total Ib Lot Accumulator", 0.0) for i, m in enumerate(all_fold_metrics) if "buy" in m)
    metrics_buy_overall = calculate_metrics(log_wf_buy, final_eq_buy, eq_buy_series_final.to_dict(), initial_capital, f"Overall WF Buy ({fund_name})", model_type_l1_used_in_run, model_type_l2_used_in_run, {"fund_profile": fund_profile}, total_ib_lot_buy_run)
    cost_keys_to_sum = ["total_commission", "total_spread", "total_slippage", "orders_blocked_dd", "orders_blocked_cooldown", "orders_scaled_lot", "be_sl_triggered_count", "tsl_triggered_count", "orders_skipped_ml_l1", "orders_skipped_ml_l2", "reentry_trades_opened", "forced_entry_trades_opened", "orders_blocked_new_v46"]
    for cost_key in cost_keys_to_sum:
        metrics_buy_overall[f"Overall WF Buy ({fund_name}) {cost_key.replace('_', ' ').title()}"] = sum(m["buy"].get(f"Fold {i+1} Buy ({fund_name}) Costs {cost_key.replace('_', ' ').title()}", 0) for i, m in enumerate(all_fold_metrics) if "buy" in m)
    metrics_buy_overall[f"Overall WF Buy ({fund_name}) Drift Overrides Active (Folds)"] = sum(m["buy"].get(f"Fold {i+1} Buy ({fund_name}) Drift Override Active", False) for i, m in enumerate(all_fold_metrics) if "buy" in m)
    metrics_buy_overall[f"Overall WF Buy ({fund_name}) Folds Ended In Recovery"] = sum(1 for i, m in enumerate(all_fold_metrics) if m.get("buy", {}).get(f"Fold {i+1} Buy ({fund_name}) Final Risk Mode") == "recovery")
    del log_wf_buy, eq_buy_series_final
    gc.collect()

    metrics_sell_overall = None
    log_wf_sell = trade_log_wf[trade_log_wf["side"] == "SELL"].copy() if not trade_log_wf.empty else pd.DataFrame()
    final_eq_sell = eq_sell_series_final.iloc[-1] if not eq_sell_series_final.empty else initial_capital
    total_ib_lot_sell_run = sum(m["sell"].get(f"Fold {i+1} Sell ({fund_name}) Costs Total Ib Lot Accumulator", 0.0) for i, m in enumerate(all_fold_metrics) if "sell" in m)
    metrics_sell_overall = calculate_metrics(log_wf_sell, final_eq_sell, eq_sell_series_final.to_dict(), initial_capital, f"Overall WF Sell ({fund_name})", model_type_l1_used_in_run, model_type_l2_used_in_run, {"fund_profile": fund_profile}, total_ib_lot_sell_run)
    for cost_key in cost_keys_to_sum:
        metrics_sell_overall[f"Overall WF Sell ({fund_name}) {cost_key.replace('_', ' ').title()}"] = sum(m["sell"].get(f"Fold {i+1} Sell ({fund_name}) Costs {cost_key.replace('_', ' ').title()}", 0) for i, m in enumerate(all_fold_metrics) if "sell" in m)
    metrics_sell_overall[f"Overall WF Sell ({fund_name}) Drift Overrides Active (Folds)"] = sum(m["sell"].get(f"Fold {i+1} Sell ({fund_name}) Drift Override Active", False) for i, m in enumerate(all_fold_metrics) if "sell" in m)
    metrics_sell_overall[f"Overall WF Sell ({fund_name}) Folds Ended In Recovery"] = sum(1 for i, m in enumerate(all_fold_metrics) if m.get("sell", {}).get(f"Fold {i+1} Sell ({fund_name}) Final Risk Mode") == "recovery")
    del log_wf_sell, eq_sell_series_final
    gc.collect()

    df_walk_forward_results_pd_final = pd.DataFrame()
    if all_fold_results_df:
        try:
            logging.info("      Combining fold result DataFrames...")
            df_walk_forward_results_pd_final = pd.concat(all_fold_results_df, axis=0, sort=False)
            del all_fold_results_df
            gc.collect()
            rows_before_dedup_final = len(df_walk_forward_results_pd_final)
            df_walk_forward_results_pd_final = df_walk_forward_results_pd_final[~df_walk_forward_results_pd_final.index.duplicated(keep='last')]
            rows_after_dedup_final = len(df_walk_forward_results_pd_final)
            if rows_before_dedup_final > rows_after_dedup_final:
                logging.info(f"      Removed {rows_before_dedup_final - rows_after_dedup_final} duplicate indices from combined results (keeping last).")
            df_walk_forward_results_pd_final.sort_index(inplace=True)
            logging.info(f"      [Runner {run_label}] (Success) Combined Final M1 Results DF. Shape: {df_walk_forward_results_pd_final.shape}")
        except Exception as e:
            logging.error(f"      [Runner {run_label}] (Error) Failed to combine Final Fold Results DF: {e}", exc_info=True)
            df_walk_forward_results_pd_final = pd.DataFrame()

    logging.info(f"      [Runner {run_label}] Returning aggregated results.")
    return (
        metrics_buy_overall, metrics_sell_overall,
        df_walk_forward_results_pd_final, trade_log_wf,
        all_equity_histories, all_fold_metrics,
        first_fold_test_data,
        model_type_l1_used_in_run,
        model_type_l2_used_in_run,
        total_ib_lot_accumulator_run
    )

logging.info("Part 9: Walk-Forward Orchestration & Analysis Functions Loaded.")
# === END OF PART 9/12 ===
# === START OF PART 10/12 ===

# ==============================================================================
# === PART 10: Main Execution & Pipeline Control (v4.8.3 Patch 1) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Implemented logging, added docstrings/comments, enhanced auto-train logic, fixed SyntaxError, added memory cleanup, added dtype passing >>>
# <<< Includes fixes from v4.7.8: Fixed missing output_dir argument in PREPARE_TRAIN_DATA backtest call >>>
# <<< MODIFIED v4.7.9 (Post-Error): Removed redundant global declaration in __main__ block >>>
# <<< MODIFIED v4.8.1: Refined auto-train logic (log loading, context cols), confirmed dtype passing, verified model loading checks, added more robust function call checks >>>
# <<< MODIFIED v4.8.2: Corrected SyntaxError in __main__ block (added except/finally for the main try block), updated log messages and versioning, robust global access in finally >>>
# <<< MODIFIED v4.8.3: Applied SyntaxError fix for try-except global variable checks to all relevant globals in this part. >>>
import logging
import os
import sys
import time
import pandas as pd
import shutil # For file moving in pipeline mode
import traceback
from joblib import load # For loading models
import gc # For memory management

# Ensure global configurations are accessible if run independently
# Define defaults if globals are not found
DEFAULT_OUTPUT_DIR = "./output_default"
DEFAULT_META_CLASSIFIER_PATH = "meta_classifier.pkl"
DEFAULT_SPIKE_MODEL_PATH = "meta_classifier_spike.pkl"
DEFAULT_CLUSTER_MODEL_PATH = "meta_classifier_cluster.pkl"
DEFAULT_FUND_NAME = "NORMAL"
DEFAULT_MODEL_TO_LINK = "catboost"
DEFAULT_ENABLE_OPTUNA_TUNING = False
DEFAULT_SAMPLE_SIZE = 60000
DEFAULT_FEATURES_TO_DROP = None
DEFAULT_MULTI_FUND_MODE = True
DEFAULT_FUND_PROFILES = {}
DEFAULT_TRAIN_META_MODEL_BEFORE_RUN = True
DEFAULT_META_CLASSIFIER_FEATURES = []
DEFAULT_RECOVERY_MODE_CONSECUTIVE_LOSSES = 4
DEFAULT_TIMEFRAME_MINUTES_M15 = 15
DEFAULT_DRIFT_WASSERSTEIN_THRESHOLD = 0.1
DEFAULT_DRIFT_TTEST_ALPHA = 0.05
DEFAULT_INITIAL_CAPITAL = 100.0
DEFAULT_N_WALK_FORWARD_SPLITS = 5
DEFAULT_OUTPUT_BASE_DIR = "/content/drive/MyDrive/new"
DEFAULT_OUTPUT_DIR_NAME = "outputgpt_v4.8.2" # Note: This might be updated by Part 1 if run
DEFAULT_DATA_FILE_PATH_M15 = "/content/drive/MyDrive/new/XAUUSD_M15.csv"
DEFAULT_DATA_FILE_PATH_M1 = "/content/drive/MyDrive/new/XAUUSD_M1.csv"
DEFAULT_META_META_CLASSIFIER_PATH = "meta_meta_classifier.pkl"
DEFAULT_USE_META_CLASSIFIER = True
DEFAULT_META_MIN_PROBA_THRESH = 0.55
DEFAULT_REENTRY_MIN_PROBA_THRESH = 0.55
DEFAULT_USE_META_META_CLASSIFIER = False
DEFAULT_META_META_MIN_PROBA_THRESH = 0.55
DEFAULT_OPTUNA_N_TRIALS = 50
DEFAULT_OPTUNA_CV_SPLITS = 5
DEFAULT_OPTUNA_METRIC = "AUC"
DEFAULT_OPTUNA_DIRECTION = "maximize"
DEFAULT_USE_GPU_ACCELERATION = True
DEFAULT_ENABLE_FORCED_ENTRY = True
DEFAULT_FORCED_ENTRY_BAR_THRESHOLD = 100
DEFAULT_FORCED_ENTRY_MIN_SIGNAL_SCORE = 0.5
DEFAULT_FORCED_ENTRY_LOOKBACK_PERIOD = 500
DEFAULT_FORCED_ENTRY_CHECK_MARKET_COND = True
DEFAULT_FORCED_ENTRY_MAX_ATR_MULT = 2.5
DEFAULT_FORCED_ENTRY_MIN_GAIN_Z_ABS = 1.0
DEFAULT_FORCED_ENTRY_ALLOWED_REGIMES = ["Normal", "Breakout", "StrongTrend"]
DEFAULT_FE_ML_FILTER_THRESHOLD = 0.40
DEFAULT_MIN_SIGNAL_SCORE_ENTRY = 2.0
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_MAX_DRAWDOWN_THRESHOLD = 0.30
DEFAULT_ENABLE_PARTIAL_TP = True
DEFAULT_PARTIAL_TP_LEVELS = [{"r_multiple": 0.8, "close_pct": 0.5}]
DEFAULT_PARTIAL_TP_MOVE_SL_TO_ENTRY = True
DEFAULT_ENABLE_KILL_SWITCH = True
DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD = 0.20
DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 7
DEFAULT_forced_entry_max_consecutive_losses = 2
DEFAULT_min_equity_threshold_pct = 0.70
DEFAULT_IB_COMMISSION_PER_LOT = 7.0
DEFAULT_EARLY_STOPPING_ROUNDS = 200
DEFAULT_CATBOOST_GPU_RAM_PART = 0.95
DEFAULT_SHAP_IMPORTANCE_THRESHOLD = 0.01
DEFAULT_PERMUTATION_IMPORTANCE_THRESHOLD = 0.001

try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR
try:
    META_CLASSIFIER_PATH
except NameError:
    META_CLASSIFIER_PATH = DEFAULT_META_CLASSIFIER_PATH
try:
    SPIKE_MODEL_PATH
except NameError:
    SPIKE_MODEL_PATH = DEFAULT_SPIKE_MODEL_PATH
try:
    CLUSTER_MODEL_PATH
except NameError:
    CLUSTER_MODEL_PATH = DEFAULT_CLUSTER_MODEL_PATH
try:
    DEFAULT_FUND_NAME
except NameError:
    DEFAULT_FUND_NAME = DEFAULT_FUND_NAME
try:
    DEFAULT_MODEL_TO_LINK
except NameError:
    DEFAULT_MODEL_TO_LINK = DEFAULT_MODEL_TO_LINK
try:
    ENABLE_OPTUNA_TUNING
except NameError:
    ENABLE_OPTUNA_TUNING = DEFAULT_ENABLE_OPTUNA_TUNING
try:
    sample_size
except NameError:
    sample_size = DEFAULT_SAMPLE_SIZE
try:
    features_to_drop
except NameError:
    features_to_drop = DEFAULT_FEATURES_TO_DROP
try:
    MULTI_FUND_MODE
except NameError:
    MULTI_FUND_MODE = DEFAULT_MULTI_FUND_MODE
try:
    FUND_PROFILES
except NameError:
    FUND_PROFILES = DEFAULT_FUND_PROFILES
try:
    TRAIN_META_MODEL_BEFORE_RUN
except NameError:
    TRAIN_META_MODEL_BEFORE_RUN = DEFAULT_TRAIN_META_MODEL_BEFORE_RUN
try:
    META_CLASSIFIER_FEATURES
except NameError:
    META_CLASSIFIER_FEATURES = DEFAULT_META_CLASSIFIER_FEATURES
try:
    RECOVERY_MODE_CONSECUTIVE_LOSSES
except NameError:
    RECOVERY_MODE_CONSECUTIVE_LOSSES = DEFAULT_RECOVERY_MODE_CONSECUTIVE_LOSSES
try:
    TIMEFRAME_MINUTES_M15
except NameError:
    TIMEFRAME_MINUTES_M15 = DEFAULT_TIMEFRAME_MINUTES_M15
try:
    DRIFT_WASSERSTEIN_THRESHOLD
except NameError:
    DRIFT_WASSERSTEIN_THRESHOLD = DEFAULT_DRIFT_WASSERSTEIN_THRESHOLD
try:
    DRIFT_TTEST_ALPHA
except NameError:
    DRIFT_TTEST_ALPHA = DEFAULT_DRIFT_TTEST_ALPHA
try:
    INITIAL_CAPITAL
except NameError:
    INITIAL_CAPITAL = DEFAULT_INITIAL_CAPITAL
try:
    N_WALK_FORWARD_SPLITS
except NameError:
    N_WALK_FORWARD_SPLITS = DEFAULT_N_WALK_FORWARD_SPLITS
try:
    OUTPUT_BASE_DIR
except NameError:
    OUTPUT_BASE_DIR = DEFAULT_OUTPUT_BASE_DIR
try:
    OUTPUT_DIR_NAME
except NameError:
    OUTPUT_DIR_NAME = DEFAULT_OUTPUT_DIR_NAME
try:
    DATA_FILE_PATH_M15
except NameError:
    DATA_FILE_PATH_M15 = DEFAULT_DATA_FILE_PATH_M15
try:
    DATA_FILE_PATH_M1
except NameError:
    DATA_FILE_PATH_M1 = DEFAULT_DATA_FILE_PATH_M1
try:
    META_META_CLASSIFIER_PATH
except NameError:
    META_META_CLASSIFIER_PATH = DEFAULT_META_META_CLASSIFIER_PATH
try:
    USE_META_CLASSIFIER
except NameError:
    USE_META_CLASSIFIER = DEFAULT_USE_META_CLASSIFIER
try:
    META_MIN_PROBA_THRESH
except NameError:
    META_MIN_PROBA_THRESH = DEFAULT_META_MIN_PROBA_THRESH
try:
    REENTRY_MIN_PROBA_THRESH
except NameError:
    REENTRY_MIN_PROBA_THRESH = DEFAULT_REENTRY_MIN_PROBA_THRESH
try:
    USE_META_META_CLASSIFIER
except NameError:
    USE_META_META_CLASSIFIER = DEFAULT_USE_META_META_CLASSIFIER
try:
    META_META_MIN_PROBA_THRESH
except NameError:
    META_META_MIN_PROBA_THRESH = DEFAULT_META_META_MIN_PROBA_THRESH
try:
    OPTUNA_N_TRIALS
except NameError:
    OPTUNA_N_TRIALS = DEFAULT_OPTUNA_N_TRIALS
try:
    OPTUNA_CV_SPLITS
except NameError:
    OPTUNA_CV_SPLITS = DEFAULT_OPTUNA_CV_SPLITS
try:
    OPTUNA_METRIC
except NameError:
    OPTUNA_METRIC = DEFAULT_OPTUNA_METRIC
try:
    OPTUNA_DIRECTION
except NameError:
    OPTUNA_DIRECTION = DEFAULT_OPTUNA_DIRECTION
try:
    USE_GPU_ACCELERATION
except NameError:
    USE_GPU_ACCELERATION = DEFAULT_USE_GPU_ACCELERATION
try:
    ENABLE_FORCED_ENTRY
except NameError:
    ENABLE_FORCED_ENTRY = DEFAULT_ENABLE_FORCED_ENTRY
try:
    FORCED_ENTRY_BAR_THRESHOLD
except NameError:
    FORCED_ENTRY_BAR_THRESHOLD = DEFAULT_FORCED_ENTRY_BAR_THRESHOLD
try:
    FORCED_ENTRY_MIN_SIGNAL_SCORE
except NameError:
    FORCED_ENTRY_MIN_SIGNAL_SCORE = DEFAULT_FORCED_ENTRY_MIN_SIGNAL_SCORE
try:
    FORCED_ENTRY_LOOKBACK_PERIOD
except NameError:
    FORCED_ENTRY_LOOKBACK_PERIOD = DEFAULT_FORCED_ENTRY_LOOKBACK_PERIOD
try:
    FORCED_ENTRY_CHECK_MARKET_COND
except NameError:
    FORCED_ENTRY_CHECK_MARKET_COND = DEFAULT_FORCED_ENTRY_CHECK_MARKET_COND
try:
    FORCED_ENTRY_MAX_ATR_MULT
except NameError:
    FORCED_ENTRY_MAX_ATR_MULT = DEFAULT_FORCED_ENTRY_MAX_ATR_MULT
try:
    FORCED_ENTRY_MIN_GAIN_Z_ABS
except NameError:
    FORCED_ENTRY_MIN_GAIN_Z_ABS = DEFAULT_FORCED_ENTRY_MIN_GAIN_Z_ABS
try:
    FORCED_ENTRY_ALLOWED_REGIMES
except NameError:
    FORCED_ENTRY_ALLOWED_REGIMES = DEFAULT_FORCED_ENTRY_ALLOWED_REGIMES
try:
    FE_ML_FILTER_THRESHOLD
except NameError:
    FE_ML_FILTER_THRESHOLD = DEFAULT_FE_ML_FILTER_THRESHOLD
try:
    MIN_SIGNAL_SCORE_ENTRY
except NameError:
    MIN_SIGNAL_SCORE_ENTRY = DEFAULT_MIN_SIGNAL_SCORE_ENTRY
try:
    DEFAULT_RISK_PER_TRADE
except NameError:
    DEFAULT_RISK_PER_TRADE = DEFAULT_RISK_PER_TRADE
try:
    MAX_DRAWDOWN_THRESHOLD
except NameError:
    MAX_DRAWDOWN_THRESHOLD = DEFAULT_MAX_DRAWDOWN_THRESHOLD
try:
    ENABLE_PARTIAL_TP
except NameError:
    ENABLE_PARTIAL_TP = DEFAULT_ENABLE_PARTIAL_TP
try:
    PARTIAL_TP_LEVELS
except NameError:
    PARTIAL_TP_LEVELS = DEFAULT_PARTIAL_TP_LEVELS
try:
    PARTIAL_TP_MOVE_SL_TO_ENTRY
except NameError:
    PARTIAL_TP_MOVE_SL_TO_ENTRY = DEFAULT_PARTIAL_TP_MOVE_SL_TO_ENTRY
try:
    ENABLE_KILL_SWITCH
except NameError:
    ENABLE_KILL_SWITCH = DEFAULT_ENABLE_KILL_SWITCH
try:
    KILL_SWITCH_MAX_DD_THRESHOLD
except NameError:
    KILL_SWITCH_MAX_DD_THRESHOLD = DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD
try:
    KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD
except NameError:
    KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD
try:
    forced_entry_max_consecutive_losses
except NameError:
    forced_entry_max_consecutive_losses = DEFAULT_forced_entry_max_consecutive_losses
try:
    min_equity_threshold_pct
except NameError:
    min_equity_threshold_pct = DEFAULT_min_equity_threshold_pct
try:
    IB_COMMISSION_PER_LOT
except NameError:
    IB_COMMISSION_PER_LOT = DEFAULT_IB_COMMISSION_PER_LOT
try:
    early_stopping_rounds_config
except NameError:
    early_stopping_rounds_config = DEFAULT_EARLY_STOPPING_ROUNDS
try:
    catboost_gpu_ram_part
except NameError:
    catboost_gpu_ram_part = DEFAULT_CATBOOST_GPU_RAM_PART
try:
    shap_importance_threshold
except NameError:
    shap_importance_threshold = DEFAULT_SHAP_IMPORTANCE_THRESHOLD
try:
    permutation_importance_threshold
except NameError:
    permutation_importance_threshold = DEFAULT_PERMUTATION_IMPORTANCE_THRESHOLD
try:
    ENTRY_CONFIG_PER_FOLD # Referenced in main
except NameError:
    ENTRY_CONFIG_PER_FOLD = DEFAULT_ENTRY_CONFIG_PER_FOLD
try:
    pattern_label_map # Referenced in main
except NameError:
    pattern_label_map = {} # Default empty dict
try:
    drift_observer # Referenced in main
except NameError:
    drift_observer = None
try:
    fold_specific_l1_thresholds # Referenced in main
except NameError:
    fold_specific_l1_thresholds = None
try:
    fold_specific_l2_thresholds # Referenced in main
except NameError:
    fold_specific_l2_thresholds = None
try:
    tuning_mode_used # Referenced in main
except NameError:
    tuning_mode_used = "Fixed Params"


# --- Auto-Train Trigger Function ---
def ensure_model_files_exist(output_dir, base_trade_log_path, base_m1_data_path):
    """
    Checks if required model files (main, spike, cluster) and their corresponding
    feature lists exist in the output directory. If any are missing, it triggers
    the training process for those specific models using the provided base data paths.

    Args:
        output_dir (str): The directory where models and features should be saved/found.
        base_trade_log_path (str): The base path (without extension) for the trade log file
                                   used for training (e.g., "trade_log_v32_walkforward").
                                   The function will look for .csv and .csv.gz.
        base_m1_data_path (str): The base path (without extension) for the M1 data file
                                 used for training (e.g., "final_data_m1_v32_walkforward").
                                 The function will look for .csv and .csv.gz.
    """
    logging.info("\n--- (Auto-Train Check) Ensuring Model Files Exist ---")
    models_to_check = {
        'main': META_CLASSIFIER_PATH,
        'spike': SPIKE_MODEL_PATH,
        'cluster': CLUSTER_MODEL_PATH,
    }
    training_needed_purposes = []

    for model_purpose, model_filename in models_to_check.items():
        model_path = os.path.join(output_dir, model_filename)
        features_filename = f"features_{model_purpose}.json"
        features_path = os.path.join(output_dir, features_filename)
        model_exists = os.path.exists(model_path)
        features_exist = os.path.exists(features_path)
        if not model_exists or not features_exist:
            if not model_exists: logging.warning(f"   (Missing) Model file for '{model_purpose}' not found: {model_path}")
            if not features_exist: logging.warning(f"   (Missing) Features file for '{model_purpose}' not found: {features_path}")
            training_needed_purposes.append(model_purpose)
        else:
            logging.info(f"   (Found) Model and Features files for '{model_purpose}' exist.")

    if not training_needed_purposes:
        logging.info("   (Success) All required model and feature files exist. No auto-training needed.")
        return

    logging.warning(f"\n   --- Triggering Auto-Training for Missing Models: {training_needed_purposes} ---")

    logging.info("      Loading base data for training...")
    trade_log_df_base = None
    train_m1_path = None

    try:
        train_log_path_gz = base_trade_log_path + ".csv.gz"
        train_log_path_csv = base_trade_log_path + ".csv"
        train_log_path = None
        if os.path.exists(train_log_path_gz):
            train_log_path = train_log_path_gz
            logging.info(f"      Found standard trade log (gz): {os.path.basename(train_log_path)}")
        elif os.path.exists(train_log_path_csv):
            train_log_path = train_log_path_csv
            logging.info(f"      Found standard trade log (csv): {os.path.basename(train_log_path)}")
        else:
            logging.warning(f"      (Info) Standard trade log ('{os.path.basename(base_trade_log_path)}.csv[.gz]') not found. Checking for fallback (prep_data)...")
            fallback_gz = os.path.join(output_dir, f"trade_log_v32_walkforward_prep_data_{DEFAULT_FUND_NAME}.csv.gz")
            fallback_csv = os.path.join(output_dir, f"trade_log_v32_walkforward_prep_data_{DEFAULT_FUND_NAME}.csv")
            if os.path.exists(fallback_gz):
                train_log_path = fallback_gz
                logging.info(f"      [Fallback] Using prep_data trade log (gz): {os.path.basename(train_log_path)}")
            elif os.path.exists(fallback_csv):
                train_log_path = fallback_csv
                logging.info(f"      [Fallback] Using prep_data trade log (csv): {os.path.basename(train_log_path)}")
            else:
                checked_paths = f"Checked: {train_log_path_csv}, {train_log_path_gz}, {fallback_csv}, {fallback_gz}"
                logging.critical(f"      (Error) Base trade log not found in standard or fallback paths. {checked_paths}")
                raise FileNotFoundError("Required trade log file for training not found.")

        trade_log_df_base = safe_load_csv_auto(train_log_path)
        if trade_log_df_base is None:
            raise ValueError(f"Failed to load trade log from: {train_log_path}")
        if trade_log_df_base.empty:
            logging.warning("      (Warning) Loaded trade log for auto-training is empty. Training will be skipped.")
            return

        logging.info(f"      (Success) Loaded trade log for training ({len(trade_log_df_base)} rows).")

        logging.debug("      Processing base trade log for training...")
        time_cols_log = ["entry_time", "close_time", "BE_Triggered_Time"]
        for col in time_cols_log:
            if col in trade_log_df_base.columns:
                trade_log_df_base[col] = pd.to_datetime(trade_log_df_base[col], errors='coerce')
        if "entry_time" not in trade_log_df_base.columns: raise ValueError("Base trade log missing 'entry_time'")
        rows_before_drop = len(trade_log_df_base)
        trade_log_df_base.dropna(subset=["entry_time"], inplace=True)
        if len(trade_log_df_base) < rows_before_drop:
            logging.warning(f"         Dropped {rows_before_drop - len(trade_log_df_base)} rows with invalid entry_time from base log.")

        context_cols_needed = {'cluster': 0, 'spike_score': 0.0, 'model_tag': 'N/A'}
        for col, default_val in context_cols_needed.items():
            if col not in trade_log_df_base.columns:
                logging.warning(f"      (Warning) Adding placeholder '{col}' column (default: {default_val}) to base trade log for auto-train.")
                trade_log_df_base[col] = default_val
        logging.info(f"      Processed Base Trade Log ({len(trade_log_df_base)} rows).")

        m1_path_std_gz = base_m1_data_path + ".csv.gz"
        m1_path_std_csv = base_m1_data_path + ".csv"
        m1_fallback_gz = os.path.join(output_dir, f"final_data_m1_v32_walkforward_prep_data_{DEFAULT_FUND_NAME}.csv.gz")
        m1_fallback_csv = os.path.join(output_dir, f"final_data_m1_v32_walkforward_prep_data_{DEFAULT_FUND_NAME}.csv")

        if os.path.exists(m1_path_std_gz):
            train_m1_path = m1_path_std_gz
            logging.info(f"      Found standard M1 data (gz): {os.path.basename(train_m1_path)}")
        elif os.path.exists(m1_path_std_csv):
            train_m1_path = m1_path_std_csv
            logging.info(f"      Found standard M1 data (csv): {os.path.basename(train_m1_path)}")
        elif os.path.exists(m1_fallback_gz):
            train_m1_path = m1_fallback_gz
            logging.warning(f"      [Fallback] Using prep_data M1 data (gz): {os.path.basename(train_m1_path)}")
        elif os.path.exists(m1_fallback_csv):
            train_m1_path = m1_fallback_csv
            logging.warning(f"      [Fallback] Using prep_data M1 data (csv): {os.path.basename(train_m1_path)}")
        else:
            checked_paths = f"Checked: {m1_path_std_csv}, {m1_path_std_gz}, {m1_fallback_csv}, {m1_fallback_gz}"
            logging.critical(f"      (Error) Base M1 data path not found in standard or fallback paths. {checked_paths}")
            raise FileNotFoundError("Required M1 data file for training not found.")
        logging.info(f"      Using M1 Data Path for Training: {train_m1_path}")

    except FileNotFoundError as fnf_error:
        logging.critical(f"      (Error) Required data file not found: {fnf_error}")
        logging.critical("         Skipping auto-training due to missing data.")
        return
    except Exception as e_load_base:
        logging.error(f"      (Error) Failed to load or process base data for auto-training: {e_load_base}", exc_info=True)
        logging.error("         Skipping auto-training.")
        return

    global features_to_drop
    for model_purpose in training_needed_purposes:
        logging.info(f"\n      --- Auto-Training Model: {model_purpose.upper()} ---")
        trade_log_filtered = None

        try:
            if model_purpose == 'spike':
                if 'spike_score' in trade_log_df_base.columns:
                    spike_threshold_train = 0.6
                    trade_log_filtered = trade_log_df_base[trade_log_df_base['spike_score'] > spike_threshold_train].copy()
                    logging.info(f"         Filtering log for Spike model (spike_score > {spike_threshold_train}): {len(trade_log_filtered)} rows")
                else:
                    logging.warning("         (Warning) 'spike_score' column not found in trade log. Cannot filter for Spike model training. Skipping.")
                    continue
            elif model_purpose == 'cluster':
                if 'cluster' in trade_log_df_base.columns:
                    cluster_train_value = 2
                    trade_log_filtered = trade_log_df_base[trade_log_df_base['cluster'] == cluster_train_value].copy()
                    logging.info(f"         Filtering log for Cluster model (cluster == {cluster_train_value}): {len(trade_log_filtered)} rows")
                else:
                    logging.warning("         (Warning) 'cluster' column not found in trade log. Cannot filter for Cluster model training. Skipping.")
                    continue
            elif model_purpose == 'main':
                trade_log_filtered = trade_log_df_base.copy()
                logging.info("         Using full log for Main model training.")
            else:
                logging.warning(f"         (Warning) Unknown model purpose '{model_purpose}' for auto-training. Skipping.")
                continue
        except Exception as e_filter:
            logging.error(f"      (Error) Failed to filter trade log for '{model_purpose}': {e_filter}", exc_info=True)
            continue

        if trade_log_filtered is None or trade_log_filtered.empty:
            logging.warning(f"         (Warning) No data available after filtering for '{model_purpose}' model. Skipping training.")
            continue

        try:
            saved_paths, _ = train_and_export_meta_model(
                trade_log_path=None,
                m1_data_path=train_m1_path,
                output_dir=output_dir,
                model_purpose=model_purpose,
                trade_log_df_override=trade_log_filtered,
                model_type_to_train="catboost",
                link_model_as_default=DEFAULT_MODEL_TO_LINK,
                enable_dynamic_feature_selection=True,
                feature_selection_method='shap',
                shap_importance_threshold=shap_importance_threshold,
                permutation_importance_threshold=permutation_importance_threshold,
                enable_optuna_tuning=False,
                sample_size=sample_size,
                features_to_drop_before_train=features_to_drop,
                early_stopping_rounds=early_stopping_rounds_config
            )
            if saved_paths is None:
                logging.warning(f"         (Warning) Auto-training for '{model_purpose}' returned None (likely skipped due to empty log inside train function).")
            elif model_purpose not in saved_paths:
                logging.error(f"         (Error) Auto-training for '{model_purpose}' completed but did not save the model file as expected.")
            else:
                logging.info(f"         (Success) Auto-training for '{model_purpose}' completed and saved.")
        except NameError as ne:
            logging.critical(f"      (CRITICAL) NameError during auto-training for '{model_purpose}': {ne}. Likely missing function definition.", exc_info=True)
            break
        except Exception as e_train:
            logging.error(f"         (Error) Exception during auto-training for '{model_purpose}': {e_train}", exc_info=True)
        finally:
            del trade_log_filtered
            gc.collect()

    del trade_log_df_base
    gc.collect()
    logging.info("--- (Auto-Train Check) Finished ---")


# --- Main Execution Function ---
def main(run_mode='FULL_PIPELINE', suffix_from_prev_step=None):
    """
    Main execution function for the Gold Trading AI script.
    Handles different run modes: PREPARE_TRAIN_DATA, TRAIN_MODEL_ONLY, FULL_RUN, FULL_PIPELINE.

    Args:
        run_mode (str): The execution mode. Defaults to 'FULL_PIPELINE'.
        suffix_from_prev_step (str, optional): Suffix passed from a previous step
                                               in FULL_PIPELINE mode. Defaults to None.

    Returns:
        str or None: The filename suffix used for the final results of the run,
                     or None if the run fails critically.
    """
    global OUTPUT_DIR, ENTRY_CONFIG_PER_FOLD, pattern_label_map
    global USE_META_CLASSIFIER, meta_model_type_used, META_MIN_PROBA_THRESH, REENTRY_MIN_PROBA_THRESH
    global USE_META_META_CLASSIFIER, meta_meta_model_type_used, META_META_MIN_PROBA_THRESH
    global M1_FEATURES_FOR_DRIFT
    global fold_specific_l1_thresholds, fold_specific_l2_thresholds, tuning_mode_used
    global INITIAL_CAPITAL, N_WALK_FORWARD_SPLITS, OUTPUT_BASE_DIR, OUTPUT_DIR_NAME
    global DATA_FILE_PATH_M15, DATA_FILE_PATH_M1, TRAIN_META_MODEL_BEFORE_RUN
    global DEFAULT_MODEL_TO_LINK, META_CLASSIFIER_PATH, META_META_CLASSIFIER_PATH
    global ENABLE_OPTUNA_TUNING, OPTUNA_N_TRIALS, OPTUNA_CV_SPLITS, OPTUNA_METRIC, OPTUNA_DIRECTION
    global META_CLASSIFIER_FEATURES
    global USE_GPU_ACCELERATION, pynvml, nvml_handle
    global ENABLE_FORCED_ENTRY, FORCED_ENTRY_BAR_THRESHOLD, FORCED_ENTRY_MIN_SIGNAL_SCORE, FORCED_ENTRY_LOOKBACK_PERIOD
    global FORCED_ENTRY_CHECK_MARKET_COND, FORCED_ENTRY_MAX_ATR_MULT, FORCED_ENTRY_MIN_GAIN_Z_ABS, FORCED_ENTRY_ALLOWED_REGIMES
    global FE_ML_FILTER_THRESHOLD
    global MIN_SIGNAL_SCORE_ENTRY, DEFAULT_RISK_PER_TRADE
    global MAX_DRAWDOWN_THRESHOLD
    global ENABLE_PARTIAL_TP, PARTIAL_TP_LEVELS, PARTIAL_TP_MOVE_SL_TO_ENTRY
    global ENABLE_KILL_SWITCH, KILL_SWITCH_MAX_DD_THRESHOLD, KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD
    global drift_observer
    global forced_entry_max_consecutive_losses, min_equity_threshold_pct
    global df_m15_dt
    global RECOVERY_MODE_CONSECUTIVE_LOSSES
    global MULTI_FUND_MODE, FUND_PROFILES, DEFAULT_FUND_NAME
    global SPIKE_MODEL_PATH, CLUSTER_MODEL_PATH
    global IB_COMMISSION_PER_LOT

    start_time_main = time.time()
    logging.info(f"\n(Starting) กำลังเริ่มฟังก์ชัน Main (Mode: {run_mode})...")
    current_run_suffix = ""

    try:
        # --- Output Directory Setup ---
        if 'OUTPUT_DIR' not in globals() or not OUTPUT_DIR:
            if 'OUTPUT_BASE_DIR' in globals() and 'OUTPUT_DIR_NAME' in globals():
                OUTPUT_DIR = setup_output_directory(OUTPUT_BASE_DIR, OUTPUT_DIR_NAME)
            else:
                logging.critical("OUTPUT_BASE_DIR or OUTPUT_DIR_NAME not defined.")
                sys.exit("ออก: ไม่สามารถกำหนด Output Directory ได้.")
        else:
            logging.info(f"   (Info) Output directory already set: {OUTPUT_DIR}")
            # Ensure the directory exists and is writable even if already set
            setup_output_directory(os.path.dirname(OUTPUT_DIR), os.path.basename(OUTPUT_DIR))

        # --- Font Setup ---
        try:
            if 'setup_fonts' in globals() and callable(setup_fonts):
                setup_fonts(OUTPUT_DIR)
            else:
                logging.warning("Function 'setup_fonts' not found. Skipping font setup.")
        except Exception as e_font:
            logging.warning(f"(Warning) เกิดข้อผิดพลาดระหว่างตั้งค่า Font: {e_font}")

        # --- GPU Utilization Check ---
        try:
            if 'print_gpu_utilization' in globals() and callable(print_gpu_utilization):
                print_gpu_utilization("After Setup")
            else:
                logging.warning("Function 'print_gpu_utilization' not found.")
        except Exception as e_gpu: # Corrected variable name for exception
            logging.warning(f"(Warning) เกิดข้อผิดพลาดระหว่างเรียก print_gpu_utilization: {e_gpu}")

    except SystemExit as e: # Catch SystemExit specifically to allow graceful exit
        logging.critical(f"(Critical Error) ข้อผิดพลาด Setup: {e}. Stopping execution.")
        return None # Or re-raise if main caller should handle it
    except Exception as e_setup:
        logging.error(f"(Error) เกิดข้อผิดพลาดระหว่าง setup: {e_setup}.", exc_info=True)
        if 'OUTPUT_DIR' not in globals() or not OUTPUT_DIR: # Check again in case it failed before assignment
            logging.critical("(Critical Error) ไม่มี OUTPUT_DIR หลังเกิดข้อผิดพลาด Setup.")
        return None # Indicate failure

    initial_intent_use_l1 = USE_META_CLASSIFIER
    local_train_model = False; local_run_final_backtest = False
    logging.info(f"   Run Mode Selected: {run_mode}")
    if run_mode == 'PREPARE_TRAIN_DATA':
        local_run_final_backtest = True
    elif run_mode == 'TRAIN_MODEL_ONLY':
        local_train_model = True
    elif run_mode == 'FULL_RUN':
        local_run_final_backtest = True
        local_train_model = TRAIN_META_MODEL_BEFORE_RUN
    elif run_mode == 'FULL_PIPELINE':
        pass
    else:
        logging.warning(f"(Warning) ไม่รู้จัก run_mode '{run_mode}'. ใช้ FULL_RUN เป็นค่าเริ่มต้น.")
        run_mode = 'FULL_RUN'; local_run_final_backtest = True; local_train_model = TRAIN_META_MODEL_BEFORE_RUN

    lag_config = {'features': ['Gain_Z', 'Candle_Speed'], 'lags': [1, 3, 5]}
    dynamic_selection_enabled = True
    selection_method = 'shap'
    shap_thresh = shap_importance_threshold
    perm_thresh = permutation_importance_threshold
    optuna_jobs_config = -1
    features_to_drop_config = features_to_drop

    if run_mode == 'FULL_PIPELINE':
        logging.info("\n(Starting) กำลังเริ่ม FULL PIPELINE...")

        logging.info("\n--- Step 1: กำลังสร้างไฟล์ Input สำหรับ Training ---")
        original_multi_fund_mode = MULTI_FUND_MODE
        MULTI_FUND_MODE = False
        logging.info("   (Pipeline Info) ปิด Multi-Fund Mode ชั่วคราว...")
        prepare_suffix = main(run_mode='PREPARE_TRAIN_DATA')
        MULTI_FUND_MODE = original_multi_fund_mode
        logging.info(f"   (Pipeline Info) คืนค่า Multi-Fund Mode: {MULTI_FUND_MODE}")
        if prepare_suffix is None:
            logging.critical("   (Error) ขั้นตอน PREPARE_TRAIN_DATA ล้มเหลว. Stopping Pipeline.")
            return None

        logging.info("\n--- Step 2: กำลังเปลี่ยนชื่อไฟล์ Input ---")
        log_file_generated_base = f"trade_log_v32_walkforward{prepare_suffix}.csv"
        data_file_generated_base = f"final_data_m1_v32_walkforward{prepare_suffix}.csv"
        log_file_generated_gz = os.path.join(OUTPUT_DIR, log_file_generated_base + ".gz")
        data_file_generated_gz = os.path.join(OUTPUT_DIR, data_file_generated_base + ".gz")
        log_file_target_base = "trade_log_v32_walkforward.csv"
        data_file_target_base = "final_data_m1_v32_walkforward.csv"
        log_file_target_gz = os.path.join(OUTPUT_DIR, log_file_target_base + ".gz")
        data_file_target_gz = os.path.join(OUTPUT_DIR, data_file_target_base + ".gz")
        rename_failed = False
        try:
            if os.path.exists(log_file_generated_gz) and os.path.exists(data_file_generated_gz):
                logging.info(f"   Moving {os.path.basename(log_file_generated_gz)} -> {os.path.basename(log_file_target_gz)}")
                if os.path.exists(log_file_target_gz): os.remove(log_file_target_gz)
                shutil.move(log_file_generated_gz, log_file_target_gz)

                logging.info(f"   Moving {os.path.basename(data_file_generated_gz)} -> {os.path.basename(data_file_target_gz)}")
                if os.path.exists(data_file_target_gz): os.remove(data_file_target_gz)
                shutil.move(data_file_generated_gz, data_file_target_gz)
                logging.info("   (Success) เปลี่ยนชื่อ/ย้ายไฟล์ (GZ) สำเร็จ.")
            else:
                logging.error(f"   (Error) ไม่พบไฟล์ GZ ที่สร้างจาก Step 1: {os.path.basename(log_file_generated_gz)} หรือ {os.path.basename(data_file_generated_gz)}. ไม่สามารถเปลี่ยนชื่อ.")
                rename_failed = True
        except Exception as e_rename:
            logging.error(f"   (Error) เกิดข้อผิดพลาดระหว่างเปลี่ยนชื่อ/ย้ายไฟล์: {e_rename}", exc_info=True)
            rename_failed = True
        if rename_failed:
            logging.critical("   (Error) หยุด FULL_PIPELINE เนื่องจากไม่สามารถ Rename ไฟล์ได้.")
            return None

        logging.info("\n--- Step 3: ตรวจสอบและ Train Models ที่ขาดหาย (Auto-Train) ---")
        try:
            if 'ensure_model_files_exist' in globals() and callable(ensure_model_files_exist):
                base_log_path_for_train = os.path.join(OUTPUT_DIR, "trade_log_v32_walkforward")
                base_m1_path_for_train = os.path.join(OUTPUT_DIR, "final_data_m1_v32_walkforward")
                ensure_model_files_exist(OUTPUT_DIR, base_log_path_for_train, base_m1_path_for_train)
            else:
                logging.critical("   (Error) Function 'ensure_model_files_exist' not found. Stopping Pipeline.")
                return None
        except Exception as e_ensure:
            logging.error(f"   (Error) เกิดข้อผิดพลาดระหว่าง ensure_model_files_exist: {e_ensure}", exc_info=True)
            logging.critical("   (Error) หยุด FULL_PIPELINE เนื่องจากไม่สามารถตรวจสอบ/Train Model.")
            return None

        logging.info("\n--- Step 4: กำลังรัน Backtest เต็มรูปแบบด้วย Models ที่มี ---")
        full_run_suffix = main(run_mode='FULL_RUN')

        logging.info("\n(Finished) FULL PIPELINE เสร็จสมบูรณ์.")
        return full_run_suffix

    final_selected_l1_features = META_CLASSIFIER_FEATURES
    drift_observer = None

    if local_train_model and run_mode == 'TRAIN_MODEL_ONLY':
        logging.info("\n(Starting) กำลัง Train Meta Classifier (L1 - Main Model Only)...")
        train_log_path_base = os.path.join(OUTPUT_DIR, "trade_log_v32_walkforward")
        train_m1_data_path_base = os.path.join(OUTPUT_DIR, "final_data_m1_v32_walkforward")
        train_log_path = train_log_path_base + ".csv.gz" if os.path.exists(train_log_path_base + ".csv.gz") else train_log_path_base + ".csv"
        train_m1_data_path = train_m1_data_path_base + ".csv.gz" if os.path.exists(train_m1_data_path_base + ".csv.gz") else train_m1_data_path_base + ".csv"

        if os.path.exists(train_log_path) and os.path.exists(train_m1_data_path):
            logging.info(f"   พบไฟล์ที่จำเป็น: Log='{os.path.basename(train_log_path)}', M1='{os.path.basename(train_m1_data_path)}'")
            try:
                train_log_df_override = safe_load_csv_auto(train_log_path)
                if train_log_df_override is None:
                    raise ValueError("Failed to load trade log for TRAIN_MODEL_ONLY.")
                if train_log_df_override.empty:
                    logging.warning("   (Warning) Trade log for training is empty. Skipping TRAIN_MODEL_ONLY.")
                    return "_train_skipped_empty_log"

                saved_paths, selected_features_from_train = train_and_export_meta_model(
                    trade_log_path=None,
                    m1_data_path=train_m1_data_path,
                    output_dir=OUTPUT_DIR,
                    model_purpose='main',
                    trade_log_df_override=train_log_df_override,
                    model_type_to_train="catboost",
                    link_model_as_default=DEFAULT_MODEL_TO_LINK,
                    enable_dynamic_feature_selection=dynamic_selection_enabled,
                    feature_selection_method=selection_method,
                    shap_importance_threshold=shap_thresh,
                    permutation_importance_threshold=perm_thresh,
                    enable_optuna_tuning=ENABLE_OPTUNA_TUNING,
                    sample_size=sample_size,
                    features_to_drop_before_train=features_to_drop_config,
                    early_stopping_rounds=early_stopping_rounds_config
                )
                if saved_paths is not None and selected_features_from_train:
                    final_selected_l1_features = selected_features_from_train
                    logging.info(f"   (Success) Train Model L1 (Main) สำเร็จ. Final Features saved to features_main.json")
                elif saved_paths is not None:
                    logging.warning("   (Warning) Train Model L1 (Main) สำเร็จ แต่ไม่ได้ Final Features List.")
                else:
                    logging.error("   (Error) Train Model L1 (Main) ล้มเหลว."); return None
                del train_log_df_override
                gc.collect()
            except NameError as ne:
                logging.critical(f"   (CRITICAL) NameError during TRAIN_MODEL_ONLY: {ne}. Likely missing function definition.", exc_info=True)
                return None
            except Exception as e_train_meta:
                logging.error(f"   (Error) เกิดข้อผิดพลาดระหว่าง Train Meta Model (L1 - Main): {e_train_meta}", exc_info=True)
                return None
        else:
            logging.error(f"   (Error) ไม่พบไฟล์ Log ('{os.path.basename(train_log_path)}') หรือ M1 Data ('{os.path.basename(train_m1_data_path)}'). ไม่สามารถ Train Model.")
            return None
        current_run_suffix = "_train_only"

    df_m1_final = None
    if run_mode in ['FULL_RUN', 'PREPARE_TRAIN_DATA']:
        logging.info(f"\n--- (Loading) กำลังโหลดและเตรียมข้อมูลสำหรับ {run_mode} ---")
        try:
            m15_dtypes = {'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32'}
            m1_dtypes = {'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32'}
            df_m15_raw = load_data(DATA_FILE_PATH_M15, "M15", dtypes=m15_dtypes)
            df_m1_raw = load_data(DATA_FILE_PATH_M1, "M1", dtypes=m1_dtypes)

            df_m15_dt = prepare_datetime(df_m15_raw, "M15")
            df_m1_dt = prepare_datetime(df_m1_raw, "M1")
            if df_m15_dt is None or df_m1_dt is None or df_m15_dt.empty or df_m1_dt.empty:
                logging.critical("(Error) ข้อมูล M15/M1 ว่างเปล่าหลัง prepare_datetime.")
                sys.exit("ออก: ข้อมูล M15/M1 ว่างเปล่าหลัง prepare_datetime.")

            df_m15_trend = calculate_m15_trend_zone(df_m15_dt)
            df_m1_features = engineer_m1_features(df_m1_dt, lag_features_config=lag_config)
            if df_m1_features is None or df_m1_features.empty:
                logging.critical("(Error) M1 ว่างเปล่าหลัง engineer_m1_features.")
                sys.exit("ออก: M1 ว่างเปล่าหลัง engineer_m1_features.")

            df_m1_cleaned, m1_features_drift_list_local = clean_m1_data(df_m1_features)
            M1_FEATURES_FOR_DRIFT = m1_features_drift_list_local
            if df_m1_cleaned is None or df_m1_cleaned.empty:
                logging.critical("(Error) M1 ว่างเปล่าหลัง clean_m1_data.")
                sys.exit("ออก: M1 ว่างเปล่าหลัง clean_m1_data.")

            logging.info("(Processing) กำลังรวม M15 Trend Zone...");
            if not isinstance(df_m1_cleaned.index, pd.DatetimeIndex): df_m1_cleaned.index = pd.to_datetime(df_m1_cleaned.index, errors='coerce'); df_m1_cleaned = df_m1_cleaned[df_m1_cleaned.index.notna()]
            if not isinstance(df_m15_trend.index, pd.DatetimeIndex): df_m15_trend.index = pd.to_datetime(df_m15_trend.index, errors='coerce'); df_m15_trend = df_m15_trend[df_m15_trend.index.notna()]
            df_m1_cleaned = df_m1_cleaned.sort_index(); df_m15_trend = df_m15_trend.sort_index()
            df_m1_merged = pd.merge_asof(df_m1_cleaned, df_m15_trend[["Trend_Zone"]], left_index=True, right_index=True, direction="backward", tolerance=pd.Timedelta(minutes=TIMEFRAME_MINUTES_M15 * 2))
            initial_trend_nan = df_m1_merged["Trend_Zone"].isna().sum();
            if initial_trend_nan > 0:
                logging.debug(f"   Filling {initial_trend_nan} NaN values in Trend_Zone with 'NEUTRAL'.")
                df_m1_merged["Trend_Zone"].fillna("NEUTRAL", inplace=True)

            logging.info("(Processing) กำลังคำนวณ M1 Entry Signals...");
            base_signal_cfg = ENTRY_CONFIG_PER_FOLD.get(0, {})
            df_m1_merged_with_signals = calculate_m1_entry_signals(df_m1_merged, base_signal_cfg)
            if df_m1_merged_with_signals.empty:
                logging.critical("(Error) M1 ว่างเปล่าหลังคำนวณ Signal.")
                sys.exit("ออก: M1 ว่างเปล่าหลังคำนวณ Signal.")

            context_cols_needed_main = ['cluster', 'spike_score', 'session', 'model_tag']
            for ccol in context_cols_needed_main:
                if ccol not in df_m1_merged_with_signals.columns:
                    logging.critical(f"   (Error) Context column '{ccol}' is missing after feature engineering!")
                    sys.exit(f"ออก: ขาด Context column ที่จำเป็น: {ccol}")

            logging.info("(Processing) กำลังกำหนดคอลัมน์ Simulation ที่จำเป็น...");
            essential_sim_cols_main = [
                "Open", "High", "Low", "Close", "ATR_14_Shifted", "Trend_Zone", "Gain_Z",
                "MACD_hist", "MACD_hist_smooth", "Candle_Speed", "Pattern_Label",
                "Entry_Long", "Entry_Short", "Trade_Tag", "Signal_Score", "Trade_Reason",
                "ATR_14", "ATR_14_Rolling_Avg", 'Volatility_Index', 'ADX', 'RSI',
                "Wick_Ratio", "Candle_Body", "Candle_Range", "Gain",
                'cluster', 'spike_score', 'session', 'model_tag'
            ]
            initial_features_list = load_features_for_model('main', OUTPUT_DIR)
            if initial_features_list is None:
                logging.warning("   (Warning) ไม่สามารถโหลด features_main.json. ใช้ Global META_CLASSIFIER_FEATURES เป็น Fallback.")
                initial_features_list = META_CLASSIFIER_FEATURES

            all_needed_features = set(essential_sim_cols_main) | set(initial_features_list)
            if lag_config:
                lag_cols_possible = [f"{feat}_lag{l}" for feat in lag_config.get('features', []) for l in lag_config.get('lags', [])]
                all_needed_features.update(lag_cols_possible)
            essential_cols_for_final_check = sorted(list(all_needed_features))
            logging.debug(f"   รวมคอลัมน์ที่อาจจำเป็นทั้งหมด: {len(essential_cols_for_final_check)}")

            logging.info("(Processing) กำลัง Drop NaN ครั้งสุดท้าย...");
            cols_present_for_dropna = [c for c in essential_cols_for_final_check if c in df_m1_merged_with_signals.columns]
            missing_for_dropna = [c for c in essential_cols_for_final_check if c not in cols_present_for_dropna]
            if missing_for_dropna:
                logging.warning(f"   (Warning) ขาดคอลัมน์สำหรับการตรวจสอบ NaN สุดท้าย: {missing_for_dropna}.")

            initial_rows = df_m1_merged_with_signals.shape[0];
            df_m1_final = df_m1_merged_with_signals.dropna(subset=cols_present_for_dropna).copy()
            rows_dropped = initial_rows - df_m1_final.shape[0]
            if rows_dropped > 0: logging.info(f"   ลบ {rows_dropped} แถวเพิ่มเติมจาก Final NaN Drop.")
            logging.info(f"   ขนาดข้อมูล M1 สุดท้าย: {df_m1_final.shape}")
            if df_m1_final.empty:
                logging.critical("(Error) Final M1 DataFrame ว่างเปล่าหลัง Drop NaN.")
                sys.exit("ออก: Final M1 DataFrame ว่างเปล่าหลัง Drop NaN.")

            logging.info("   [RAM Opt] กำลังแปลง Final M1 Data เป็น float32...")
            final_numeric_cols = df_m1_final.select_dtypes(include=np.number).columns
            converted_count = 0
            for col in final_numeric_cols:
                if col == 'cluster': continue
                if pd.api.types.is_float_dtype(df_m1_final[col].dtype):
                    try:
                        df_m1_final[col] = pd.to_numeric(df_m1_final[col], downcast='float')
                        if df_m1_final[col].dtype == 'float32': converted_count += 1
                    except Exception as e_astype_final:
                        logging.warning(f"      (Warning) ไม่สามารถแปลงคอลัมน์ '{col}' เป็น float32: {e_astype_final}. ใช้ float64 ต่อไป.")
            logging.info(f"   (Success) แปลง {converted_count} Final M1 Data columns เป็น float32 (เท่าที่ทำได้) สำเร็จ.")

            final_check_cols_present = [c for c in cols_present_for_dropna if c in df_m1_final.columns]
            nan_check_final = df_m1_final[final_check_cols_present].isnull().any()
            if nan_check_final.any():
                missing_cols_log = nan_check_final[nan_check_final].index.tolist()
                logging.critical(f"(Error) พบ NaN หลัง DropNA ในคอลัมน์: {missing_cols_log}")
                sys.exit("   ออก: พบ NaN ที่ไม่คาดคิดในข้อมูล Final M1.")
            else:
                logging.info("   (Success) Final M1 Data ผ่านการตรวจสอบ NaN.")

            logging.debug("   Cleaning up intermediate dataframes after data preparation...")
            del df_m15_raw, df_m1_raw, df_m15_dt, df_m1_dt, df_m15_trend
            del df_m1_features, df_m1_cleaned, df_m1_merged, df_m1_merged_with_signals
            gc.collect()
            logging.debug("   Intermediate dataframe cleanup complete.")

            if run_mode == 'PREPARE_TRAIN_DATA':
                logging.info("\n--- (Run & Save) Running Backtest and Saving PREPARE_TRAIN_DATA results ---")
                suffix = f"_prep_data_{DEFAULT_FUND_NAME}"
                current_run_suffix = suffix
                try:
                    logging.info(f"   Running initial backtest for Fund: {DEFAULT_FUND_NAME} to generate trade log...")
                    prep_fund_profile = FUND_PROFILES.get(DEFAULT_FUND_NAME, {"risk": DEFAULT_RISK_PER_TRADE, "mm_mode": "balanced"})
                    prep_fund_profile['name'] = DEFAULT_FUND_NAME

                    (
                        _, _,
                        _, prep_trade_log_wf,
                        _, _,
                        _,
                        _, _,
                        _
                    ) = run_all_folds_with_threshold(
                        fund_profile=prep_fund_profile,
                        current_l1_threshold=META_MIN_PROBA_THRESH,
                        df_m1_final=df_m1_final,
                        available_models=None,
                        model_switcher_func=None,
                        output_dir=OUTPUT_DIR,
                    )

                    m1_save_path = os.path.join(OUTPUT_DIR, f"final_data_m1_v32_walkforward{suffix}.csv.gz")
                    logging.info(f"   Saving final M1 data to: {m1_save_path}")
                    df_m1_final.to_csv(m1_save_path, index=True, encoding="utf-8", compression="gzip")
                    logging.info(f"   (Success) Saved final M1 data: {os.path.basename(m1_save_path)}")

                    log_save_path = os.path.join(OUTPUT_DIR, f"trade_log_v32_walkforward{suffix}.csv.gz")
                    logging.info(f"   Saving generated trade log to: {log_save_path}")
                    if prep_trade_log_wf is not None and not prep_trade_log_wf.empty:
                        prep_trade_log_wf.to_csv(log_save_path, index=False, encoding="utf-8", compression="gzip")
                        logging.info(f"   (Success) Saved generated trade log ({len(prep_trade_log_wf)} rows): {os.path.basename(log_save_path)}")
                    else:
                        logging.warning(f"   (Warning) Backtest for PREPARE_TRAIN_DATA generated an empty or None trade log. Saving empty log file.")
                        pd.DataFrame(columns=["entry_time"]).to_csv(log_save_path, index=False, encoding="utf-8", compression="gzip")

                    logging.info(f"(Finished) PREPARE_TRAIN_DATA ran backtest and saved results -> suffix={suffix}")
                    del df_m1_final, prep_trade_log_wf
                    gc.collect()
                    return current_run_suffix
                except NameError as ne:
                    logging.critical(f"   (CRITICAL) NameError during PREPARE_TRAIN_DATA backtest: {ne}. Likely missing function definition.", exc_info=True)
                    return None
                except Exception as e_prep_run_save:
                    logging.error(f"   (Error) Failed to run backtest or save PREPARE_TRAIN_DATA results: {e_prep_run_save}", exc_info=True)
                    return None

        except SystemExit as se:
            logging.critical(f"(Critical Error) ข้อผิดพลาดการเตรียมข้อมูล: {se}. Stopping execution.")
            return None
        except AssertionError as ae:
            logging.critical(f"(Critical Error) Data Validation Failed: {ae}. Stopping execution.")
            return None
        except NameError as ne:
            logging.critical(f"(CRITICAL) NameError during data preparation: {ne}. Likely missing function definition.", exc_info=True)
            return None
        except Exception as e_data_prep:
            logging.critical(f"(Error) ข้อผิดพลาดร้ายแรงระหว่างเตรียมข้อมูล: {e_data_prep}", exc_info=True)
            return None
    else:
        df_m1_final = None
        logging.info(f"(Info) ข้ามการโหลดและเตรียมข้อมูล (Mode: {run_mode})")

    if run_mode == 'FULL_RUN':
        try:
            logging.info(f"\n--- ({run_mode}) ตรวจสอบ/Train Models ที่ขาดหาย ---")
            if 'ensure_model_files_exist' in globals() and callable(ensure_model_files_exist):
                base_log_path_for_train = os.path.join(OUTPUT_DIR, "trade_log_v32_walkforward")
                base_m1_path_for_train = os.path.join(OUTPUT_DIR, "final_data_m1_v32_walkforward")
                ensure_model_files_exist(OUTPUT_DIR, base_log_path_for_train, base_m1_path_for_train)
            else:
                logging.critical("   (Error) Function 'ensure_model_files_exist' not found. Cannot auto-train.")
                return None
        except Exception as e_ensure:
            logging.error(f"   (Error) เกิดข้อผิดพลาดระหว่าง ensure_model_files_exist: {e_ensure}", exc_info=True)
            logging.error("   (Error) หยุดการทำงานเนื่องจากไม่สามารถตรวจสอบ/Train Model.")
            return None

    available_models = {}
    if run_mode == 'FULL_RUN':
        logging.info("\n--- กำลังโหลด Models และ Features สำหรับ FULL_RUN (Model Switcher) ---")
        model_keys = ['main', 'spike', 'cluster']
        model_paths = {
            "main": os.path.join(OUTPUT_DIR, META_CLASSIFIER_PATH),
            "spike": os.path.join(OUTPUT_DIR, SPIKE_MODEL_PATH),
            "cluster": os.path.join(OUTPUT_DIR, CLUSTER_MODEL_PATH),
        }

        for model_key in model_keys:
            model_path = model_paths[model_key]
            logging.info(f"(Loading) พยายามโหลด Model '{model_key}' จาก: {model_path}")
            loaded_model = None
            if not os.path.exists(model_path):
                logging.error(f"  (Error) ไม่พบไฟล์ Model '{model_key}' ({os.path.basename(model_path)}).")
                if model_key == 'main':
                    logging.critical("   (CRITICAL) Main model file is missing. Cannot proceed with FULL_RUN.")
                    return None
            else:
                try:
                    loaded_model = load(model_path)
                    model_type = loaded_model.__class__.__name__
                    if "CatBoostClassifier" not in model_type:
                        logging.error(f"  (Error) Model '{model_key}' is not a CatBoostClassifier (Type: {model_type}).")
                        loaded_model = None
                    elif not hasattr(loaded_model, 'predict_proba'):
                        logging.error(f"  (Error) Model '{model_key}' does not have 'predict_proba' method.")
                        loaded_model = None
                    else:
                        logging.info(f"  (Success) โหลด Model '{model_key}' ({model_type}) สำเร็จ.")
                except Exception as e:
                    logging.error(f"  (Error) ไม่สามารถโหลด Model '{model_key}': {e}", exc_info=True)
                    loaded_model = None
                    if model_key == 'main':
                        logging.critical("   (CRITICAL) Failed to load main model. Cannot proceed.")
                        return None

            logging.info(f"(Loading) พยายามโหลด Features สำหรับ '{model_key}'...")
            features_list = load_features_for_model(model_key, OUTPUT_DIR)
            if features_list is None:
                logging.error(f"  (Error) ไม่สามารถโหลด Features สำหรับ Model '{model_key}'.")
                if loaded_model is not None:
                    logging.warning(f"      (Invalidating) Model '{model_key}' ถูกปิดใช้งานเนื่องจากโหลด Features ไม่สำเร็จ.")
                    loaded_model = None
                if model_key == 'main':
                    logging.critical("   (CRITICAL) Failed to load features for main model. Cannot proceed.")
                    return None
            else:
                logging.info(f"  (Success) โหลด Features ({len(features_list)}) สำหรับ Model '{model_key}' สำเร็จ.")

            available_models[model_key] = {'model': loaded_model, 'features': features_list if features_list else []}

        if available_models.get('main', {}).get('model') is None:
            logging.critical("(Critical Error) ไม่สามารถโหลด Main Model หรือ Features ได้. หยุดการทำงาน.")
            return None

        logging.info(f"--- โหลด Models/Features เสร็จสิ้น: {[k for k, v in available_models.items() if v.get('model') is not None]} ---")
        USE_META_CLASSIFIER = available_models.get('main', {}).get('model') is not None
        USE_META_META_CLASSIFIER = False

    drift_observer = None
    if df_m1_final is not None and run_mode != 'TRAIN_MODEL_ONLY':
        if M1_FEATURES_FOR_DRIFT:
            try:
                drift_observer = DriftObserver(M1_FEATURES_FOR_DRIFT)
            except NameError:
                logging.warning("Class 'DriftObserver' not found. Skipping drift analysis.")
                drift_observer = None
        else:
            logging.warning("(Warning) M1_FEATURES_FOR_DRIFT ว่างเปล่า. ไม่สามารถสร้าง DriftObserver.")

    tuning_mode_used = "Fixed Params"
    logging.info(f"\n(Info) ข้าม Auto Threshold Tuning (ใช้ {tuning_mode_used} สำหรับ Model).")
    best_l1_threshold_final = META_MIN_PROBA_THRESH;
    fold_specific_l1_thresholds = None; fold_specific_l2_thresholds = None

    final_run_suffix = "_skipped"
    all_funds_metrics_buy = {}
    all_funds_metrics_sell = {}
    all_funds_df_results = {}
    all_funds_trade_logs = {}
    all_funds_equity_histories = {}
    all_funds_fold_metrics = {}
    total_ib_lot_all_funds = 0.0

    if run_mode == 'FULL_RUN' and df_m1_final is not None:
        funds_to_run = {}
        if MULTI_FUND_MODE:
            funds_to_run = FUND_PROFILES
            logging.info(f"\n(Multi-Fund Mode) กำลังรันสำหรับ {len(funds_to_run)} Fund Profiles: {list(funds_to_run.keys())}")
        else:
            default_profile = FUND_PROFILES.get(DEFAULT_FUND_NAME, {"risk": DEFAULT_RISK_PER_TRADE, "mm_mode": "balanced"})
            funds_to_run = {DEFAULT_FUND_NAME: default_profile}
            logging.info(f"\n(Single Fund Mode) กำลังรันสำหรับ Fund Profile: {DEFAULT_FUND_NAME}")

        for fund_name, fund_profile_config in funds_to_run.items():
            fund_profile_config['name'] = fund_name
            logging.info("\n" + "=" * 20 + f" STARTING FUND: {fund_name} (MM Mode: {fund_profile_config.get('mm_mode', 'N/A')}) " + "=" * 20)

            ml_used_in_run = USE_META_CLASSIFIER
            ml_status_suffix = "switcher_on" if ml_used_in_run else "ml_off"
            tuning_status_suffix = "fixed_params"
            final_run_suffix_fund = f"_{fund_name}_{tuning_status_suffix}_{ml_status_suffix}"
            current_run_suffix = final_run_suffix_fund

            logging.info(f"--- FINAL WALK-FORWARD RUN (Fund: {fund_name}, Suffix: {final_run_suffix_fund}) ---")

            final_l1_override = best_l1_threshold_final;
            available_models_run = available_models
            try:
                model_switcher_func_run = select_model_for_trade
            except NameError:
                logging.critical("   (CRITICAL) Function 'select_model_for_trade' not found. Cannot run with model switching.")
                return None

            if available_models_run is None or available_models_run.get('main', {}).get('model') is None:
                logging.critical(f"    (CRITICAL ERROR) Main model became None before run_all_folds for fund '{fund_name}'! Stopping.")
                if not MULTI_FUND_MODE: return None
                else: continue
            if not callable(model_switcher_func_run):
                logging.critical(f"    (CRITICAL ERROR) model_switcher_func_run is not callable for fund '{fund_name}'! Stopping.")
                if not MULTI_FUND_MODE: return None
                else: continue

            try:
                (
                    metrics_buy_overall_fund, metrics_sell_overall_fund,
                    df_walk_forward_results_pd_fund, trade_log_wf_fund,
                    all_equity_histories_fund, all_fold_metrics_fund,
                    first_fold_test_data_for_shap_final,
                    model_type_l1_from_sim_final, model_type_l2_from_sim_final,
                    total_ib_lot_fund
                ) = run_all_folds_with_threshold(
                    fund_profile=fund_profile_config,
                    current_l1_threshold=final_l1_override,
                    df_m1_final=df_m1_final,
                    available_models=available_models_run,
                    model_switcher_func=model_switcher_func_run,
                    drift_observer=drift_observer,
                    output_dir=OUTPUT_DIR,
                )
            except NameError as ne_orch:
                logging.critical(f"   (CRITICAL) NameError during final backtest run: {ne_orch}. Likely missing function definition.", exc_info=True)
                return None
            except Exception as e_orch:
                logging.critical(f"   (CRITICAL) Error during final backtest run for fund '{fund_name}': {e_orch}", exc_info=True)
                if not MULTI_FUND_MODE: return None
                else: continue

            logging.info(f"\n(Aggregating) กำลังรวมผลลัพธ์ Final Walk-Forward Run (Fund: {fund_name})...")
            if metrics_buy_overall_fund is None or metrics_sell_overall_fund is None:
                logging.error(f"(Error) Final Walk-Forward Run (Fund: {fund_name}) ล้มเหลว หรือไม่ได้สร้าง Metrics.")
                if not MULTI_FUND_MODE: return current_run_suffix
                else: continue
            else:
                all_funds_metrics_buy[fund_name] = metrics_buy_overall_fund
                all_funds_metrics_sell[fund_name] = metrics_sell_overall_fund
                all_funds_df_results[fund_name] = df_walk_forward_results_pd_fund
                all_funds_trade_logs[fund_name] = trade_log_wf_fund
                all_funds_equity_histories[fund_name] = all_equity_histories_fund
                all_funds_fold_metrics[fund_name] = all_fold_metrics_fund
                total_ib_lot_all_funds += total_ib_lot_fund

                if fund_name == DEFAULT_FUND_NAME and USE_META_CLASSIFIER and available_models.get('main', {}).get('model') and first_fold_test_data_for_shap_final is not None:
                    logging.info("\n--- SHAP Analysis (Final Run - Validation Set Data - Main Model) ---")
                    main_model_obj = available_models['main']['model']
                    main_model_features = load_features_for_model('main', OUTPUT_DIR)
                    if main_model_features and first_fold_test_data_for_shap_final is not None:
                        try:
                            if 'analyze_feature_importance_shap' in globals() and callable(analyze_feature_importance_shap):
                                analyze_feature_importance_shap(main_model_obj, main_model_obj.__class__.__name__, first_fold_test_data_for_shap_final, main_model_features, OUTPUT_DIR, fold_idx=None)
                            else:
                                logging.warning("   Function 'analyze_feature_importance_shap' not found. Skipping SHAP analysis.")
                        except Exception as e_shap_final:
                            logging.error(f"   Error during final SHAP analysis: {e_shap_final}", exc_info=True)
                    else:
                        logging.warning("   (Warning) ไม่สามารถโหลด features_main.json หรือ first_fold_test_data ว่างเปล่า สำหรับ SHAP analysis.")
                    del first_fold_test_data_for_shap_final
                    gc.collect()

                if drift_observer and fund_name == list(funds_to_run.keys())[0]:
                    logging.info("\n--- Drift Summary (Final Run - Overall) ---")
                    try:
                        if drift_observer is not None:
                            drift_observer.summarize_and_save(OUTPUT_DIR)
                        else:
                            logging.warning("   (Warning) drift_observer is None. Skipping drift summary.")
                    except Exception as e_drift_sum:
                        logging.error(f"   Error summarizing/saving drift results: {e_drift_sum}", exc_info=True)

                if df_walk_forward_results_pd_fund is not None and not df_walk_forward_results_pd_fund.empty:
                    logging.info(f"\n--- Saving Results for Fund: {fund_name} (Suffix: {final_run_suffix_fund}) ---")
                    metrics_all_fund = {**metrics_buy_overall_fund, **metrics_sell_overall_fund}
                    logging.info(f"\n--- Metrics Summary (Fund: {fund_name}) ---")
                    for k, v in metrics_all_fund.items(): logging.info(f"   {k}: {v}")
                    logging.info("------------------------------------")

                    metrics_file_path = os.path.join(OUTPUT_DIR, f"metrics_summary_v32{final_run_suffix_fund}.csv")
                    try:
                        pd.DataFrame([metrics_all_fund]).T.to_csv(metrics_file_path, header=False, encoding="utf-8")
                        logging.info(f"   (Success) Saved Metrics Summary: {metrics_file_path}")
                    except Exception as e:
                        logging.error(f"   (Error) Failed to save metrics summary: {e}", exc_info=True)

                    log_file_path = os.path.join(OUTPUT_DIR, f"trade_log_v32_walkforward{final_run_suffix_fund}.csv")
                    try:
                        trade_log_wf_fund.to_csv(log_file_path + ".gz", index=False, encoding="utf-8", compression="gzip")
                        logging.info(f"   (Success) Saved Trade Log (GZ): {log_file_path}.gz")
                    except Exception as e_gz:
                        logging.warning(f"   (Warning) Failed to save trade log as GZ: {e_gz}. Attempting CSV...")
                        try:
                            trade_log_wf_fund.to_csv(log_file_path, index=False, encoding="utf-8")
                            logging.info(f"   (Success) Saved Trade Log (CSV - Fallback): {log_file_path}")
                        except Exception as e_csv:
                            logging.error(f"   (Error) Failed to save trade log (CSV): {e_csv}", exc_info=True)

                    try:
                        fold_boundaries = [df_m1_final.index.min()] + [df_m1_final.iloc[test_index].index.max() for _, test_index in tscv.split(df_m1_final)]
                        eq_buy_hist_fund_plot_dict = all_funds_equity_histories[fund_name].get(f"Fold0_BUY_{fund_name}", {})
                        eq_sell_hist_fund_plot_dict = all_funds_equity_histories[fund_name].get(f"Fold0_SELL_{fund_name}", {})
                        eq_buy_hist_fund_plot = pd.Series(dict(sorted(eq_buy_hist_fund_plot_dict.items()))).sort_index()
                        eq_buy_hist_fund_plot = eq_buy_hist_fund_plot[~eq_buy_hist_fund_plot.index.duplicated(keep='last')]
                        eq_sell_hist_fund_plot = pd.Series(dict(sorted(eq_sell_hist_fund_plot_dict.items()))).sort_index()
                        eq_sell_hist_fund_plot = eq_sell_hist_fund_plot[~eq_sell_hist_fund_plot.index.duplicated(keep='last')]

                        if 'plot_equity_curve' in globals() and callable(plot_equity_curve):
                            plot_equity_curve(eq_buy_hist_fund_plot, f"Equity Curve - BUY ({fund_name})", initial_capital, OUTPUT_DIR, f"buy{final_run_suffix_fund}", fold_boundaries)
                            plot_equity_curve(eq_sell_hist_fund_plot, f"Equity Curve - SELL ({fund_name})", initial_capital, OUTPUT_DIR, f"sell{final_run_suffix_fund}", fold_boundaries)
                        else:
                            logging.warning("   Function 'plot_equity_curve' not found. Skipping equity plots.")
                    except Exception as e_plot:
                        logging.error(f"   Error generating equity plots for fund '{fund_name}': {e_plot}", exc_info=True)

                    final_data_path = os.path.join(OUTPUT_DIR, f"final_data_m1_v32_walkforward{final_run_suffix_fund}.csv")
                    try:
                        logging.info(f"   Saving Final M1 Data with results to: {final_data_path}.gz")
                        df_walk_forward_results_pd_fund.to_csv(final_data_path + ".gz", index=True, encoding="utf-8", compression="gzip")
                        logging.info(f"   (Success) Saved Final M1 Data (GZ): {final_data_path}.gz")
                    except Exception as e_gz:
                        logging.warning(f"   (Warning) Failed to save final M1 data as GZ: {e_gz}. Attempting CSV...")
                        try:
                            df_walk_forward_results_pd_fund.to_csv(final_data_path, index=True, encoding="utf-8")
                            logging.info(f"   (Success) Saved Final M1 Data (CSV - Fallback): {final_data_path}")
                        except Exception as e_csv:
                            logging.error(f"   (Error) Failed to save final M1 data (CSV): {e_csv}", exc_info=True)
                else:
                    logging.error(f"(Error) Final Walk-forward (Fund: {fund_name}) ไม่ได้สร้างผลลัพธ์รวม (df_walk_forward_results_pd_fund is empty or None).")
                del df_walk_forward_results_pd_fund, trade_log_wf_fund, all_equity_histories_fund, all_fold_metrics_fund
                gc.collect()

        if MULTI_FUND_MODE and run_mode == 'FULL_RUN' and len(funds_to_run) > 1:
            logging.info("\n" + "=" * 20 + " MULTI-FUND RUN COMPLETED " + "=" * 20)
            if all_funds_trade_logs:
                logging.info("   Combining trade logs from all funds...")
                all_funds_combined_log = pd.concat(all_funds_trade_logs.values(), ignore_index=True)
                combined_log_path = os.path.join(OUTPUT_DIR, f"trade_log_v32_walkforward_ALL_FUNDS.csv")
                try:
                    all_funds_combined_log.to_csv(combined_log_path + ".gz", index=False, encoding="utf-8", compression="gzip")
                    logging.info(f"   (Success) Saved Combined Trade Log (GZ): {combined_log_path}.gz")
                    del all_funds_combined_log
                    gc.collect()
                except Exception as e_comb_log:
                    logging.error(f"   (Error) Failed to save combined trade log: {e_comb_log}", exc_info=True)
            else:
                logging.warning("   (Warning) No trade logs found to combine for ALL_FUNDS summary.")
            logging.info("\n--- IB Commission Summary (All Funds) ---")
            logging.info(f"   Total Accumulated Lots (All Funds): {total_ib_lot_all_funds:.2f}")
            logging.info(f"   Estimated Total IB Commission (USD): ${total_ib_lot_all_funds * IB_COMMISSION_PER_LOT:.2f}")
            logging.info("-----------------------------------------")

        if run_mode == 'FULL_RUN':
            final_run_suffix = final_run_suffix_fund if not MULTI_FUND_MODE or len(funds_to_run) == 1 else "_MultiFund_Run"
        elif run_mode == 'PREPARE_TRAIN_DATA':
            final_run_suffix = current_run_suffix

        if 'df_m1_final' in locals() and df_m1_final is not None: del df_m1_final
        gc.collect()

    elif run_mode == 'TRAIN_MODEL_ONLY':
        logging.info("\n(Info) ข้าม Final Walk-Forward Run (Mode: TRAIN_MODEL_ONLY).")
        final_run_suffix = current_run_suffix
    elif run_mode == 'PREPARE_TRAIN_DATA':
        logging.info("\n(Info) ข้าม Final Walk-Forward Run (Mode: PREPARE_TRAIN_DATA - files saved earlier).")
        final_run_suffix = current_run_suffix
    elif df_m1_final is None and run_mode != 'TRAIN_MODEL_ONLY':
        logging.warning("\n(Info) ข้าม Final Walk-Forward Run (ไม่มีข้อมูล M1 Final).")
        final_run_suffix = "_no_data"

    if USE_GPU_ACCELERATION and pynvml and nvml_handle:
        try:
            logging.info("Attempting to shut down pynvml...")
            if 'print_gpu_utilization' in globals() and callable(print_gpu_utilization): print_gpu_utilization("Final State")
            pynvml.nvmlShutdown()
            logging.info("(Success) ปิดการทำงาน pynvml สำเร็จ.")
        except Exception as e:
            logging.warning(f"(Warning) เกิดข้อผิดพลาดขณะปิด pynvml: {e}")

    end_time_main = time.time()
    logging.info(f"\n--- ฟังก์ชัน Main (Mode: {run_mode}) เสร็จสิ้นใน {end_time_main - start_time_main:.2f} วินาที ---")

    return final_run_suffix


# ==============================================================================
# === SCRIPT ENTRY POINT ===
# ==============================================================================
if __name__ == "__main__":
    start_time_script = time.time()
    logging.info(f"(Starting) Script Gold Trading AI v4.8.2...") # Updated version

    selected_run_mode = 'FULL_PIPELINE'
    # selected_run_mode = 'PREPARE_TRAIN_DATA'
    # selected_run_mode = 'TRAIN_MODEL_ONLY'
    # selected_run_mode = 'FULL_RUN'

    logging.info(f"(Starting) กำลังเริ่มการทำงานหลัก (main) ในโหมด: {selected_run_mode}...")
    final_run_suffix = None
    # <<< MODIFIED v4.8.2: Ensured this try...except...finally block is correctly structured >>>
    try:
        # Initialize variables that might be used in finally or are good to have defaults for
        fold_specific_l1_thresholds = None
        fold_specific_l2_thresholds = None
        tuning_mode_used = "Fixed Params"
        drift_observer = None
        df_m15_dt = None
        # gold_ai2025_version_tuple = (4, 8, 2) # Example, if needed for dynamic log name in finally

        final_run_suffix = main(run_mode=selected_run_mode)

        if selected_run_mode not in ['TRAIN_MODEL_ONLY', 'PREPARE_TRAIN_DATA']:
            logging.info("\n--- (Post-Run) Starting Log Analysis ---")
            if ('OUTPUT_DIR' in globals() and OUTPUT_DIR and
                final_run_suffix and final_run_suffix not in ["_skipped", "_no_data", "_train_only", "_train_skipped_empty_log"]):

                log_suffix_to_analyze = "_ALL_FUNDS" if MULTI_FUND_MODE and selected_run_mode == 'FULL_RUN' and len(FUND_PROFILES)>1 else final_run_suffix
                log_path_base = os.path.join(OUTPUT_DIR, f"trade_log_v32_walkforward{log_suffix_to_analyze}.csv")
                log_path = log_path_base + ".gz" if os.path.exists(log_path_base + ".gz") else log_path_base

                if os.path.exists(log_path):
                    logging.info(f"Analyzing log file: {log_path}")
                    try:
                        if 'run_log_analysis_pipeline' in globals() and callable(run_log_analysis_pipeline):
                            analysis_results = run_log_analysis_pipeline(
                                log_path,
                                OUTPUT_DIR,
                                RECOVERY_MODE_CONSECUTIVE_LOSSES, # Make sure this global is available
                                suffix=log_suffix_to_analyze
                            )
                            if analysis_results: logging.info("\n(Log Analysis Completed)")
                        else:
                            logging.warning("Function 'run_log_analysis_pipeline' not found. Skipping log analysis.")
                    except Exception as e_log_analysis:
                        logging.error(f"Error during log analysis: {e_log_analysis}", exc_info=True)
                else:
                    logging.warning(f"\n(Skipping Log Analysis) Log file not found: {log_path}")
            else:
                logging.warning(f"\n(Skipping Log Analysis) No valid log file suffix ('{final_run_suffix}') or OUTPUT_DIR for analysis.")
        else:
            logging.info(f"\n(Skipping Log Analysis) Run mode '{selected_run_mode}' does not require log analysis.")

    except SystemExit as se_main: # Renamed to avoid conflict
        logging.critical(f"\n(Critical Error) สคริปต์ออกก่อนเวลา: {se_main}")
        # Optionally re-raise or handle further if needed: raise se_main
    except KeyboardInterrupt:
        logging.warning("\n(Stopped) การทำงานหยุดโดยผู้ใช้ (KeyboardInterrupt).")
    except NameError as ne_main: # Renamed
        logging.critical(f"\n(Error) NameError in __main__: '{ne_main}'. Critical function or variable likely missing.", exc_info=True)
    except Exception as e_main_general: # Renamed
        logging.critical("\n(Error) เกิดข้อผิดพลาดที่ไม่คาดคิดใน __main__:", exc_info=True)
    finally:
        end_time_script = time.time()
        total_duration = end_time_script - start_time_script
        logging.info(f"\n(Finished) Script Gold Trading AI v4.8.2 เสร็จสมบูรณ์!") # Updated version
        
        final_tuning_mode_log = "Unknown"
        # Check globals first, then locals if not found in globals (though it should be global)
        if 'tuning_mode_used' in globals() and globals()['tuning_mode_used'] is not None:
            final_tuning_mode_log = globals()['tuning_mode_used']
        elif 'tuning_mode_used' in locals() and locals()['tuning_mode_used'] is not None: # Check local scope as fallback
            final_tuning_mode_log = locals()['tuning_mode_used']
        logging.info(f"   Tuning Mode ที่ใช้: {final_tuning_mode_log}")

        output_dir_final_path = None
        try:
            # Ensure these globals are accessible or defined before use
            output_base_dir_val = globals().get('OUTPUT_BASE_DIR')
            output_dir_name_val = globals().get('OUTPUT_DIR_NAME')
            output_dir_val = globals().get('OUTPUT_DIR')
            log_filename_val = globals().get('LOG_FILENAME', 'gold_ai_unknown_version.log') # Use LOG_FILENAME from top

            if output_base_dir_val and output_dir_name_val:
                output_dir_final_path = os.path.join(output_base_dir_val, output_dir_name_val)
            elif output_dir_val:
                output_dir_final_path = output_dir_val

            if output_dir_final_path and os.path.exists(output_dir_final_path):
                logging.info(f"   ผลลัพธ์ถูกบันทึกไปที่: {output_dir_final_path}")
                logging.info(f"   ไฟล์ Log หลัก: {log_filename_val}") # Directly use LOG_FILENAME
            elif output_dir_final_path:
                logging.warning(f"   (Warning) ไม่พบ Output Directory ที่คาดหวัง: {output_dir_final_path}")
            else:
                logging.warning("   (Warning) ไม่สามารถกำหนด Output Directory path.")
        except Exception as e_report_path:
            logging.warning(f"   (Warning) Error reporting output path: {e_report_path}")

        logging.info(f"   เวลาดำเนินการทั้งหมด: {total_duration:.2f} วินาที ({total_duration/60:.2f} นาที).")
        logging.info("--- End of Script ---")

# === END OF PART 10/12 ===
# === START OF PART 11/12 ===

# ==============================================================================
# === PART 11: MT5 Connector (Placeholder) (v4.8.1) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Added logging and comments for placeholder, fixed NameError in get_live_data default arg >>>
# <<< MODIFIED v4.8.1: Version bump in comments, placeholder logic remains >>>
import logging
import time
# import MetaTrader5 as mt5 # Import commented out as it's a placeholder
# import pandas as pd # Import commented out as it's not used in placeholder

logging.info("Loading Part 11: MT5 Connector (Placeholder)...")

# --- MT5 Connection Parameters (Placeholder Examples) ---
# These would typically be loaded from a secure config file or environment variables
MT5_LOGIN = 12345678       # Replace with actual account number
MT5_PASSWORD = "YOUR_PASSWORD" # Replace with actual password
MT5_SERVER = "YOUR_SERVER"   # Replace with actual server name
MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe" # Example path

# --- Placeholder Functions ---

def initialize_mt5():
    """
    Placeholder function to initialize connection to MetaTrader 5 terminal.
    (Currently does nothing but log).
    """
    logging.info("Attempting to initialize MT5 connection (Placeholder)...")
    # Example connection logic (commented out):
    # if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    #     logging.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
    #     mt5.shutdown()
    #     return False
    # else:
    #     logging.info("MT5 initialized successfully.")
    #     acc_info = mt5.account_info()
    #     if acc_info:
    #         logging.info(f"Connected to MT5 Account: {acc_info.login} on {acc_info.server}")
    #     else:
    #         logging.warning("Could not retrieve MT5 account info.")
    #     return True
    logging.warning("   MT5 connection logic is currently commented out (Placeholder).")
    return False # Return False as it's a placeholder

def shutdown_mt5():
    """
    Placeholder function to shut down the MetaTrader 5 connection.
    (Currently does nothing but log).
    """
    logging.info("Attempting to shut down MT5 connection (Placeholder)...")
    # Example shutdown logic (commented out):
    # mt5.shutdown()
    # logging.info("MT5 connection shut down.")
    pass

def get_live_data(symbol="XAUUSD", timeframe=1, count=100):
    """
    Placeholder function to get live market data from MT5.
    (Currently returns None).

    Args:
        symbol (str): Trading symbol.
        timeframe (int): MT5 timeframe constant (e.g., 1 for M1, 15 for M15). Defaults to 1.
        count (int): Number of bars to retrieve.
    """
    logging.debug(f"Attempting to get live data for {symbol} (Placeholder)...")
    # Example data fetching logic (commented out):
    # # Need to import mt5 and pandas if uncommenting
    # import MetaTrader5 as mt5
    # import pandas as pd
    # # Map integer timeframe back to mt5 constant if needed inside the function
    # mt5_timeframe_map = {1: mt5.TIMEFRAME_M1, 15: mt5.TIMEFRAME_M15, ...}
    # actual_mt5_timeframe = mt5_timeframe_map.get(timeframe, mt5.TIMEFRAME_M1) # Default to M1 if invalid
    # rates = mt5.copy_rates_from_pos(symbol, actual_mt5_timeframe, 0, count)
    # if rates is None:
    #     logging.error(f"Failed to get rates for {symbol}, error code = {mt5.last_error()}")
    #     return None
    # elif len(rates) == 0:
    #     logging.warning(f"No rates returned for {symbol} (Count: {count})")
    #     return pd.DataFrame()
    # else:
    #     df = pd.DataFrame(rates)
    #     df['time'] = pd.to_datetime(df['time'], unit='s')
    #     df.set_index('time', inplace=True)
    #     logging.info(f"Successfully retrieved {len(df)} rates for {symbol}.")
    #     return df
    return None # Return None as it's a placeholder

def execute_mt5_order(action_type=0, symbol="XAUUSD", lot_size=0.01, price=None, sl=None, tp=None, deviation=10, magic=12345):
    """
    Placeholder function to execute a market order on MT5.
    (Currently does nothing but log).

    Args:
        action_type (int): mt5.ORDER_TYPE_BUY (0) or mt5.ORDER_TYPE_SELL (1). Defaults to 0.
        symbol (str): Trading symbol.
        lot_size (float): Order volume.
        price (float, optional): Entry price (for market orders, MT5 uses current price).
        sl (float, optional): Stop loss price.
        tp (float, optional): Take profit price.
        deviation (int): Slippage deviation in points.
        magic (int): Magic number for the order.

    Returns:
        dict or None: Result of the order execution (or None if placeholder).
    """
    logging.info(f"Attempting to execute MT5 order (Placeholder): Action={action_type}, Symbol={symbol}, Lot={lot_size}, SL={sl}, TP={tp}")
    # Example order execution logic (commented out):
    # # Need to import mt5 if uncommenting
    # import MetaTrader5 as mt5
    # # Map integer action_type back to mt5 constant if needed inside the function
    # actual_mt5_action = mt5.ORDER_TYPE_BUY if action_type == 0 else mt5.ORDER_TYPE_SELL
    # request = {
    #     "action": mt5.TRADE_ACTION_DEAL,
    #     "symbol": symbol,
    #     "volume": lot_size,
    #     "type": actual_mt5_action,
    #     "price": mt5.symbol_info_tick(symbol).ask if actual_mt5_action == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
    #     "sl": sl if sl else 0.0,
    #     "tp": tp if tp else 0.0,
    #     "deviation": deviation,
    #     "magic": magic,
    #     "comment": "Python Script Order",
    #     "type_time": mt5.ORDER_TIME_GTC, # Good till cancelled
    #     "type_filling": mt5.ORDER_FILLING_IOC, # Immediate or Cancel
    # }
    # result = mt5.order_send(request)
    # if result is None:
    #     logging.error(f"MT5 order_send failed for {symbol}. Error code: {mt5.last_error()}")
    #     return None
    # elif result.retcode != mt5.TRADE_RETCODE_DONE:
    #     logging.error(f"MT5 order failed: retcode={result.retcode}, comment={result.comment}")
    #     # Log request details for debugging
    #     logging.debug(f"Failed Order Request Details: {request}")
    #     return result._asdict() # Return result dictionary on failure
    # else:
    #     logging.info(f"MT5 Order executed successfully: Deal={result.deal}, Order={result.order}")
    #     return result._asdict() # Return result dictionary on success
    logging.warning("   MT5 order execution logic is currently commented out (Placeholder).")
    return None # Return None as it's a placeholder

# --- Main Live Trading Loop (Conceptual Placeholder) ---
def run_live_trading_loop():
    """
    Conceptual placeholder for the main live trading loop.
    This would involve getting live data, calculating features/signals,
    making decisions (potentially using loaded ML models), and executing orders via MT5.
    """
    logging.info("Starting Live Trading Loop (Conceptual Placeholder)...")
    if not initialize_mt5():
        logging.critical("Cannot start live trading loop: MT5 initialization failed.")
        return

    try:
        while True: # Loop indefinitely (or based on some condition)
            logging.info("Live Loop Iteration...")
            # 1. Get Live Data
            # live_data = get_live_data("XAUUSD", timeframe=1, count=500) # Use integer timeframe
            # if live_data is None or live_data.empty:
            #     logging.warning("Could not get live data, sleeping...")
            #     time.sleep(60) # Wait before retrying
            #     continue

            # 2. Calculate Features & Signals
            # Need to adapt feature engineering functions (Part 5) for live data
            # features = engineer_m1_features(live_data) # Example
            # signals = calculate_m1_entry_signals(features, live_config) # Use live config

            # 3. Make Trading Decision
            # last_bar = features.iloc[-1]
            # decision = "HOLD"
            # if last_bar['Entry_Long'] == 1: decision = "BUY"
            # elif last_bar['Entry_Short'] == 1: decision = "SELL"

            # 4. Apply ML Filter (if enabled)
            # if USE_META_CLASSIFIER and decision != "HOLD":
            #     # Load model, select features, predict probability
            #     # Adjust decision based on probability threshold
            #     pass

            # 5. Execute Order (if decision is BUY/SELL)
            # if decision == "BUY":
            #     # Calculate lot size, SL, TP based on live conditions
            #     execute_mt5_order(action_type=0, symbol="XAUUSD", lot_size=lot, sl=sl_price, tp=tp_price) # Use integer action_type
            # elif decision == "SELL":
            #     execute_mt5_order(action_type=1, symbol="XAUUSD", lot_size=lot, sl=sl_price, tp=tp_price) # Use integer action_type

            # 6. Manage Open Positions (Check SL/TP, Trailing Stops)
            # Need functions to get open positions from MT5 and modify/close them
            # manage_open_positions()

            # 7. Wait for the next bar/interval
            logging.info("Live loop iteration complete (Placeholder). Sleeping...")
            time.sleep(60) # Example: Wait for 1 minute

    except KeyboardInterrupt:
        logging.info("Live trading loop interrupted by user.")
    except Exception as e:
        logging.critical(f"Critical error in live trading loop: {e}", exc_info=True)
    finally:
        # Ensure MT5 connection is closed properly
        shutdown_mt5()
        logging.info("Live Trading Loop Finished.")

# Note: This part remains a placeholder. Actual MT5 integration requires
# installing the MetaTrader5 library, handling authentication securely,
# adapting data fetching and order execution logic, and robust error handling.

logging.info("Part 11: MT5 Connector (Placeholder) Loaded.")
# === END OF PART 11/12 ===
# === START OF PART 12/12 ===

# ==============================================================================
# === PART 12: End of Script Marker (v4.8.1) ===
# ==============================================================================
# <<< MODIFIED v4.7.9: Updated version marker >>>
# <<< MODIFIED v4.8.1: Version bump in comments >>>
# This part serves as a clear marker for the end of the refactored script file.
# No functional code is typically placed here.
# ==============================================================================
import logging

logging.info("Reached End of Part 12 (End of Script Marker).")
# === END OF PART 12/12 ===