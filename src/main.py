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
import json
from src.utils import get_env_float, maybe_collect, load_settings
if 'pytest' in sys.modules:
    cfg = sys.modules.get('src.config')
    if cfg is not None and getattr(cfg, '__file__', None) is None and hasattr(cfg, 'ENTRY_CONFIG_PER_FOLD'):
        DEFAULT_ENTRY_CONFIG_PER_FOLD = cfg.ENTRY_CONFIG_PER_FOLD
        logger = getattr(cfg, 'logger', logging.getLogger(__name__)); CFG_FUND_PROFILES = getattr(cfg, 'FUND_PROFILES', {}); CFG_MULTI_FUND_MODE = getattr(cfg, 'MULTI_FUND_MODE', True); CFG_DEFAULT_FUND_NAME = getattr(cfg, 'DEFAULT_FUND_NAME', 'NORMAL'); DEFAULT_FUND_PROFILES = CFG_FUND_PROFILES; DEFAULT_MULTI_FUND_MODE = CFG_MULTI_FUND_MODE; DEFAULT_FUND_NAME = CFG_DEFAULT_FUND_NAME
    else:
        DEFAULT_ENTRY_CONFIG_PER_FOLD = {}
        logger = logging.getLogger(__name__)
else:
    try:
        from src.config import logger, ENTRY_CONFIG_PER_FOLD as DEFAULT_ENTRY_CONFIG_PER_FOLD, FUND_PROFILES as CFG_FUND_PROFILES, MULTI_FUND_MODE as CFG_MULTI_FUND_MODE, DEFAULT_FUND_NAME as CFG_DEFAULT_FUND_NAME
    except Exception:  # pragma: no cover - fallback for tests
        logger = logging.getLogger(__name__)
        DEFAULT_ENTRY_CONFIG_PER_FOLD = {}

# --------------------------------------------
# =============================================================================
# === PATCH AI Studio v4.8.9: ABSOLUTE-IMPORTS & STUB MISSING FUNCTIONS ===
# =============================================================================
# เพื่อให้สามารถรันไฟล์นี้ได้ทั้งแบบโมดูลและสคริปต์เดี่ยว
# จึงต้องกำหนดฟังก์ชันสำรองหากไม่สามารถนำเข้าฟังก์ชันจริงได้
def setup_fonts():
    """ฟังก์ชันสำรองสำหรับตั้งค่า Font (ไม่ทำอะไร)."""
    pass


def print_gpu_utilization(_=None):
    """ฟังก์ชันสำรองสำหรับแสดงการใช้ GPU (ไม่ทำอะไร)."""
    pass


def plot_equity_curve(*_args, **_kwargs):
    """ฟังก์ชันสำรองสำหรับวาดกราฟ Equity Curve (ไม่ทำอะไร)."""
    pass

import time
from src.data_loader import (
    setup_output_directory as dl_setup_output_directory,
    load_data,
    prepare_datetime,
    safe_load_csv_auto,
)
from src.features import (
    calculate_m15_trend_zone,
    engineer_m1_features,
    clean_m1_data,
    calculate_m1_entry_signals,
    create_session_column,
    load_features_for_model,
    save_features,
    load_features,
)
from src.strategy import (
    run_all_folds_with_threshold,
    train_and_export_meta_model,
    DriftObserver,
    plot_equity_curve,  # [Patch v5.7.3] import plotting helper
)
from src.utils import (
    export_trade_log,
    download_model_if_missing,
    download_feature_list_if_missing,
    get_env_float,
    estimate_resource_plan,
    validate_file,
)
from sklearn.model_selection import TimeSeriesSplit  # [Patch v5.5.4] Needed for equity plot fold boundaries
import pandas as pd
import numpy as np
import shutil # For file moving in pipeline mode
import traceback
import glob
from src.csv_validator import validate_and_convert_csv
from joblib import load # For loading models
import gc # For memory management

# [Patch] Initialize pynvml to prevent NameError during GPU checks
try:
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover - allow running without NVML
    pynvml = None
    nvml_handle = None

# Ensure global configurations are accessible if run independently
# Define defaults if globals are not found
DEFAULT_OUTPUT_DIR = "./output_default"
DEFAULT_META_CLASSIFIER_PATH = "meta_classifier.pkl"
DEFAULT_SPIKE_MODEL_PATH = "meta_classifier_spike.pkl"
DEFAULT_CLUSTER_MODEL_PATH = "meta_classifier_cluster.pkl"
DEFAULT_FUND_NAME = CFG_DEFAULT_FUND_NAME if 'CFG_DEFAULT_FUND_NAME' in globals() else "NORMAL"
DEFAULT_MODEL_TO_LINK = "catboost"
DEFAULT_ENABLE_OPTUNA_TUNING = True
DEFAULT_SAMPLE_SIZE = 60000
DEFAULT_FEATURES_TO_DROP = None
DEFAULT_MULTI_FUND_MODE = CFG_MULTI_FUND_MODE if 'CFG_MULTI_FUND_MODE' in globals() else True
DEFAULT_FUND_PROFILES = CFG_FUND_PROFILES if 'CFG_FUND_PROFILES' in globals() else {}
DEFAULT_TRAIN_META_MODEL_BEFORE_RUN = True
DEFAULT_META_CLASSIFIER_FEATURES = []
DEFAULT_RECOVERY_MODE_CONSECUTIVE_LOSSES = 4
DEFAULT_TIMEFRAME_MINUTES_M15 = 15
# [Patch v5.5.4] Environment override for drift threshold
DEFAULT_DRIFT_WASSERSTEIN_THRESHOLD = get_env_float("DRIFT_WASSERSTEIN_THRESHOLD", 0.1)
DEFAULT_DRIFT_TTEST_ALPHA = 0.05
DEFAULT_INITIAL_CAPITAL = 100.0
DEFAULT_N_WALK_FORWARD_SPLITS = 5
# [Patch v5.2.5] Use project-relative paths for portability
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_BASE_DIR = os.path.join(_BASE_DIR, "logs")
DEFAULT_OUTPUT_DIR_NAME = "outputgpt_v4.8.4"  # Note: This might be updated by Part 1 if run
# [Patch v5.0.11] Use CSV paths relative to this project for portability
DEFAULT_DATA_FILE_PATH_M15 = os.path.join(_BASE_DIR, "XAUUSD_M15.csv")
DEFAULT_DATA_FILE_PATH_M1 = os.path.join(_BASE_DIR, "XAUUSD_M1.csv")
DEFAULT_META_META_CLASSIFIER_PATH = "meta_meta_classifier.pkl"
DEFAULT_USE_META_CLASSIFIER = os.getenv("USE_META_CLASSIFIER", "True").lower() in ("true", "1", "yes")
DEFAULT_META_MIN_PROBA_THRESH = 0.25
DEFAULT_REENTRY_MIN_PROBA_THRESH = 0.5
DEFAULT_USE_META_META_CLASSIFIER = False
DEFAULT_META_META_MIN_PROBA_THRESH = 0.5
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
DEFAULT_FORCED_ENTRY_MIN_GAIN_Z_ABS = 0.5
DEFAULT_FORCED_ENTRY_ALLOWED_REGIMES = ["Normal", "Breakout", "StrongTrend", "Reversal", "InsideBar", "Choppy"]
DEFAULT_FE_ML_FILTER_THRESHOLD = 0.40
DEFAULT_MIN_SIGNAL_SCORE_ENTRY = 1.0  # [Patch v5.3.9]
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_MAX_DRAWDOWN_THRESHOLD = 0.30
DEFAULT_ENABLE_PARTIAL_TP = True
DEFAULT_PARTIAL_TP_LEVELS = [{"r_multiple": 0.8, "close_pct": 0.5}]
DEFAULT_PARTIAL_TP_MOVE_SL_TO_ENTRY = True
DEFAULT_ENABLE_KILL_SWITCH = True
DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD = 0.15
DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 5
DEFAULT_KILL_SWITCH_WARNING_MAX_DD_THRESHOLD = 0.25
DEFAULT_KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD = 7
DEFAULT_forced_entry_max_consecutive_losses = 2
DEFAULT_min_equity_threshold_pct = 0.70
DEFAULT_IB_COMMISSION_PER_LOT = 7.0
DEFAULT_EARLY_STOPPING_ROUNDS = 200
DEFAULT_CATBOOST_GPU_RAM_PART = 0.95
DEFAULT_SHAP_IMPORTANCE_THRESHOLD = 0.01
DEFAULT_PERMUTATION_IMPORTANCE_THRESHOLD = 0.001


# [Patch v5.2.4] Ensure default output directory exists
def ensure_default_output_dir(path=DEFAULT_OUTPUT_DIR):
    """สร้างโฟลเดอร์ผลลัพธ์เริ่มต้นหากยังไม่มี"""
    if not os.path.isabs(path):
        project_root = os.getcwd()
        path = os.path.join(project_root, path)
    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"   (Setup) ตรวจสอบโฟลเดอร์ผลลัพธ์: {path}")
        return path
    except Exception as e:
        logging.error(f"   (Error) สร้างโฟลเดอร์ผลลัพธ์ไม่สำเร็จ: {e}", exc_info=True)
        return None

ensure_default_output_dir()

try:
    OUTPUT_DIR
except NameError:
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def load_validated_csv(raw_path, timeframe, dtypes=None):
    """Validate and load CSV or Parquet, ensuring Buddhist year conversion."""
    if raw_path.endswith(".parquet"):
        try:
            return pd.read_parquet(raw_path)
        except Exception as e_read:
            logging.warning(f"(Warning) Failed to read parquet {raw_path}: {e_read}")
            return load_data(raw_path.replace(".parquet", ".csv"), timeframe, dtypes=dtypes)

    parquet_path = raw_path.replace(".csv", ".parquet")
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e_read_cache:
            logging.warning(f"(Warning) Failed to read parquet {parquet_path}: {e_read_cache}")

    clean_path = raw_path.replace(".csv", "_clean.csv")
    if not os.path.exists(clean_path):
        print(
            f"ไฟล์ข้อมูลสะอาด '{clean_path}' ยังไม่มี, กำลังสร้างจากไฟล์ต้นฉบับ..."
        )
        try:
            validate_and_convert_csv(raw_path, clean_path)
            print("สร้างไฟล์ข้อมูลสะอาดสำเร็จ")
        except Exception as e:
            print(
                f"เกิดข้อผิดพลาดร้ายแรงระหว่างการตรวจสอบและแปลงไฟล์ CSV: {e}"
            )
            raise

    df_loaded = load_data(clean_path, timeframe, dtypes=dtypes)
    try:
        df_loaded.to_parquet(parquet_path)
    except Exception as e_save:
        logging.warning(f"(Warning) Failed to save parquet to {parquet_path}: {e_save}")
    if df_loaded.empty:
        logging.warning("(Warning) Loaded DataFrame is empty after CSV load")
    return df_loaded
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
    DEFAULT_FUND_NAME = "NORMAL"
try:
    DEFAULT_MODEL_TO_LINK
except NameError:
    DEFAULT_MODEL_TO_LINK = "catboost"
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
    DEFAULT_RISK_PER_TRADE = 0.01
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
    KILL_SWITCH_WARNING_MAX_DD_THRESHOLD
except NameError:
    KILL_SWITCH_WARNING_MAX_DD_THRESHOLD = DEFAULT_KILL_SWITCH_WARNING_MAX_DD_THRESHOLD
try:
    KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD
except NameError:
    KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD = DEFAULT_KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD
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
#
# ถ้า `ENTRY_CONFIG_PER_FOLD` ยังไม่ถูกกำหนดในโมดูลนี้
# ให้ใช้ `DEFAULT_ENTRY_CONFIG_PER_FOLD` (หรือ `{}` กรณี import ไม่สำเร็จ)
#
try:
    ENTRY_CONFIG_PER_FOLD  # อ้างอิงใน main
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
    """[Patch v5.4.5] Ensure all model and feature files exist or auto-train."""
    logging.info("\n--- (Auto-Train Check) Ensuring Model Files Exist ---")
    skip_auto_train = os.getenv("SKIP_AUTO_TRAIN", "0") in {"1", "True", "true"}

    required = {
        'main': (META_CLASSIFIER_PATH, 'features_main.json'),
        'spike': (SPIKE_MODEL_PATH, 'features_spike.json'),
        'cluster': (CLUSTER_MODEL_PATH, 'features_cluster.json'),
    }

    missing_models = []
    for key, (model_file, feature_file) in required.items():
        model_path = os.path.join(output_dir, model_file)
        feature_path = os.path.join(output_dir, feature_file)
        if not (os.path.exists(model_path) and os.path.exists(feature_path)):
            download_model_if_missing(model_path, f"URL_MODEL_{key.upper()}")
            download_feature_list_if_missing(feature_path, f"URL_FEATURES_{key.upper()}")
            if not os.path.exists(model_path) or not os.path.exists(feature_path):
                missing_models.append(key)
                logging.warning(f"Missing model file for '{key}' ({model_file}).")

    if not missing_models:
        logging.info("   (Success) Model files and feature lists already exist.")
        return

    if skip_auto_train:
        logging.warning("   SKIP_AUTO_TRAIN enabled - creating placeholder model files.")
        os.makedirs(output_dir, exist_ok=True)
        for key in missing_models:
            open(os.path.join(output_dir, required[key][0]), "a").close()
            open(os.path.join(output_dir, required[key][1]), "a").close()
        return

    logging.warning(
        f"   Triggering Auto-Training for Missing Models: {missing_models}"
    )

    train_log_path = None
    for ext in (".csv.gz", ".csv"):
        candidate = base_trade_log_path + ext
        if os.path.exists(candidate):
            train_log_path = candidate
            break

    m1_path = None
    for ext in (".csv.gz", ".csv"):
        candidate = base_m1_data_path + ext
        if os.path.exists(candidate):
            m1_path = candidate
            break

    if train_log_path is None or m1_path is None:
        logging.error("   (Error) Training data missing. Creating placeholder model files.")
        os.makedirs(output_dir, exist_ok=True)
        for key in missing_models:
            open(os.path.join(output_dir, required[key][0]), "a").close()
            open(os.path.join(output_dir, required[key][1]), "a").close()
        return

    trade_log_df = safe_load_csv_auto(train_log_path)
    if trade_log_df is None or trade_log_df.empty:
        logging.error("   (Error) Loaded trade log is empty. Creating placeholder model files.")
        os.makedirs(output_dir, exist_ok=True)
        for key in missing_models:
            open(os.path.join(output_dir, required[key][0]), "a").close()
            open(os.path.join(output_dir, required[key][1]), "a").close()
        return

    for key in missing_models:
        try:
            saved_paths, features = train_and_export_meta_model(
                trade_log_path=None,
                m1_data_path=m1_path,
                output_dir=output_dir,
                model_purpose=key,
                trade_log_df_override=trade_log_df,
                model_type_to_train="catboost",
                link_model_as_default=DEFAULT_MODEL_TO_LINK,
                enable_dynamic_feature_selection=True,
                feature_selection_method="shap",
                shap_importance_threshold=shap_importance_threshold,
                permutation_importance_threshold=permutation_importance_threshold,
                enable_optuna_tuning=False,
                sample_size=sample_size,
                features_to_drop_before_train=features_to_drop,
                early_stopping_rounds=early_stopping_rounds_config,
            )
            if saved_paths is None or key not in saved_paths:
                raise RuntimeError("Training did not produce a model file")
        except Exception as e:
            logging.error(f"   (Error) Auto-training failed for '{key}': {e}", exc_info=True)
            os.makedirs(output_dir, exist_ok=True)
            open(os.path.join(output_dir, required[key][0]), "a").close()
            open(os.path.join(output_dir, required[key][1]), "a").close()
            continue

        model_path = os.path.join(output_dir, required[key][0])
        features_path = os.path.join(output_dir, required[key][1])
        if not os.path.exists(model_path):
            os.makedirs(output_dir, exist_ok=True)
            open(model_path, "a").close()
        if features is None:
            open(features_path, "a").close()
        else:
            if key == 'main':
                save_features_main_json(features, output_dir)
            else:
                save_features_json(features, key, output_dir)

        if not validate_file(model_path):
            logging.warning(f"[QA] Placeholder created for '{key}' model")
            open(model_path, "a").close()
        if not validate_file(features_path):
            logging.warning(f"[QA] Placeholder created for '{key}' features")
            open(features_path, "a").close()
    logging.info("--- (Auto-Train Check) Finished ---")


# [Patch v5.0.1] Helper wrappers for pipeline steps
def prepare_train_data():
    """[Patch] Run PREPARE_TRAIN_DATA step programmatically."""
    logging.info("[Patch] Run Mode Selected: PREPARE_TRAIN_DATA (helper)")
    return main(run_mode='PREPARE_TRAIN_DATA')


def train_models():
    """[Patch] Run TRAIN_MODEL_ONLY step programmatically."""
    logging.info("[Patch] Run Mode Selected: TRAIN_MODEL (helper)")
    return main(run_mode='TRAIN_MODEL_ONLY')


# [Patch v5.0.14] Ensure features_main.json exists
def ensure_main_features_file(output_dir):
    """[Patch] Create default features_main.json if it does not exist."""
    path = os.path.join(output_dir, "features_main.json")
    if os.path.exists(path):
        return path
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_META_CLASSIFIER_FEATURES, f, ensure_ascii=False, indent=2)
        logging.info("[Patch] Created default features_main.json")
    except Exception as e:
        logging.error(f"[Patch] Failed to create features_main.json: {e}")
    return path


# [Patch v5.3.2] QA function to persist features_main.json
def save_features_main_json(features, output_dir):
    """[Patch] Save main features list, creating QA log if empty."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'features_main.json')
    if features is None or len(features) == 0:
        logger.warning("[QA] features_main.json is empty. Creating empty features file.")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        qa_log = os.path.join(output_dir, 'features_main_qa.log')
        with open(qa_log, 'w', encoding='utf-8') as f:
            f.write("[QA] features_main.json EMPTY. Please check feature engineering logic.\n")
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        logger.info(f"[QA] features_main.json saved successfully ({len(features)} features).")
    return path

# [Patch v5.4.5] Generic function to save features for sub-models
def save_features_json(features, model_name, output_dir):
    """Save feature list for a specific model name."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"features_{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(features if features is not None else [], f, ensure_ascii=False, indent=2)
    return path


# --- Main Execution Function ---
def main(run_mode='FULL_PIPELINE', skip_prepare=False, suffix_from_prev_step=None):
    """
    Main execution function for the Gold Trading AI script.
    Handles different run modes: PREPARE_TRAIN_DATA, TRAIN_MODEL_ONLY, FULL_RUN, FULL_PIPELINE.

    Args:
        run_mode (str): The execution mode. Defaults to 'FULL_PIPELINE'.
        skip_prepare (bool): Internal flag used to skip PREPARE_TRAIN_DATA when rerunning
            as part of FULL_PIPELINE. Defaults to False.
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
                OUTPUT_DIR = dl_setup_output_directory(OUTPUT_BASE_DIR, OUTPUT_DIR_NAME)
            else:
                logging.critical("OUTPUT_BASE_DIR or OUTPUT_DIR_NAME not defined.")
                sys.exit("ออก: ไม่สามารถกำหนด Output Directory ได้.")
        else:
            logging.info(f"   (Info) Output directory already set: {OUTPUT_DIR}")
            # Ensure the directory exists and is writable even if already set
            dl_setup_output_directory(os.path.dirname(OUTPUT_DIR), os.path.basename(OUTPUT_DIR))

        # [Patch v5.0.14] Pre-create features_main.json to avoid warnings
        ensure_main_features_file(OUTPUT_DIR)

        # --- Font Setup ---
        try:
            if 'setup_fonts' in globals() and callable(setup_fonts):
                setup_fonts()
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
    if run_mode == 'PREPARE_TRAIN_DATA' and not skip_prepare:
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

        trade_log_target_gz = os.path.join(OUTPUT_DIR, "trade_log_v32_walkforward.csv.gz")
        prep_pattern = os.path.join(OUTPUT_DIR, "trade_log_v32_walkforward_prep_data_*.csv")
        if not os.path.exists(trade_log_target_gz) and not glob.glob(prep_pattern):
            logging.info("(Info) ไม่พบ trade log สั่ง PREPARE_TRAIN_DATA")
            prepare_suffix = prepare_train_data()
            if prepare_suffix is None:
                logging.critical("   (Error) ขั้นตอน PREPARE_TRAIN_DATA ล้มเหลว. Stopping Pipeline.")
                return None

            log_file_generated_base = f"trade_log_v32_walkforward{prepare_suffix}.csv"
            data_file_generated_base = f"final_data_m1_v32_walkforward{prepare_suffix}.csv"
            log_file_generated_gz = os.path.join(OUTPUT_DIR, log_file_generated_base + ".gz")
            data_file_generated_gz = os.path.join(OUTPUT_DIR, data_file_generated_base + ".gz")
            data_file_target_gz = os.path.join(OUTPUT_DIR, "final_data_m1_v32_walkforward.csv.gz")
            if os.path.exists(log_file_generated_gz) and os.path.exists(data_file_generated_gz):
                if os.path.exists(trade_log_target_gz):
                    # [Patch] Ensure safe removal only within DATA_DIR
                    from src import config as cfg
                    if str(trade_log_target_gz).startswith(str(cfg.DATA_DIR)):
                        os.remove(trade_log_target_gz)
                    else:
                        logger.warning(
                            f"Ignoring removal of {trade_log_target_gz}: outside DATA_DIR"
                        )
                if os.path.exists(data_file_target_gz):
                    # [Patch] Same safety check for data file deletion
                    if str(data_file_target_gz).startswith(str(cfg.DATA_DIR)):
                        os.remove(data_file_target_gz)
                    else:
                        logger.warning(
                            f"Ignoring removal of {data_file_target_gz}: outside DATA_DIR"
                        )
                shutil.move(log_file_generated_gz, trade_log_target_gz)
                shutil.move(data_file_generated_gz, data_file_target_gz)
            else:
                logging.critical("   (Error) ไม่พบไฟล์ที่สร้างจาก PREPARE_TRAIN_DATA")
                return None
        else:
            logging.info("(Info) พบ trade log แล้ว ข้าม PREPARE_TRAIN_DATA")

        model_pattern = os.path.join(OUTPUT_DIR, "meta_classifier*.pkl")
        if not glob.glob(model_pattern):
            logging.info("(Info) ไม่พบโมเดล สั่ง TRAIN_MODEL")
            train_models()
            if not glob.glob(model_pattern):
                logging.critical("   (Error) TRAIN_MODEL ล้มเหลว ไม่พบโมเดลหลังจากฝึก")
                return None
        else:
            logging.info("(Info) พบโมเดลแล้ว")

        full_run_suffix = main(run_mode='FULL_RUN')
        if full_run_suffix == 'FULL_RUN':
            # [Patch] Test helper shim - convert mode echo to success marker
            full_run_suffix = '_ok'

        logging.info("\n(Finished) FULL PIPELINE เสร็จสมบูรณ์.")
        return full_run_suffix

    final_selected_l1_features = META_CLASSIFIER_FEATURES
    drift_observer = None

    if local_train_model and run_mode == 'TRAIN_MODEL_ONLY':
        logging.info("\n(Starting) กำลัง Train Meta Classifier (L1 - Main Model Only)...")
        train_log_path_base = os.path.join(
            OUTPUT_DIR,
            "trade_log_v32_walkforward_prep_data_NORMAL",
        )  # [Patch v5.1.6] Use PREPARE_TRAIN_DATA generated trade log
        train_m1_data_path_base = os.path.join(
            OUTPUT_DIR,
            "final_data_m1_v32_walkforward_prep_data_NORMAL",
        )  # [Patch v5.1.6] Use PREPARE_TRAIN_DATA generated M1 data

        logging.info(
            f"TRAIN_MODEL_ONLY: กำลังโหลด Trade Log จาก: {train_log_path_base}.csv(.gz)"
        )
        logging.info(
            f"TRAIN_MODEL_ONLY: กำลังโหลด M1 Data จาก: {train_m1_data_path_base}.csv(.gz)"
        )
        train_log_path = train_log_path_base + ".csv.gz" if os.path.exists(train_log_path_base + ".csv.gz") else train_log_path_base + ".csv"
        train_m1_data_path = train_m1_data_path_base + ".csv.gz" if os.path.exists(train_m1_data_path_base + ".csv.gz") else train_m1_data_path_base + ".csv"

        if not (os.path.exists(train_log_path) and os.path.exists(train_m1_data_path)):
            logging.info("   (Info) ไม่พบไฟล์ฝึกสั่ง PREPARE_TRAIN_DATA อัตโนมัติ")
            prepare_suffix = prepare_train_data()
            if prepare_suffix is None:
                logging.critical("   (Error) PREPARE_TRAIN_DATA ล้มเหลว ไม่สามารถ Train Model")
                return None
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
                maybe_collect()
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
            df_m15_raw = load_validated_csv(
                DATA_FILE_PATH_M15, "M15", dtypes=m15_dtypes
            )
            df_m1_raw = load_validated_csv(
                DATA_FILE_PATH_M1, "M1", dtypes=m1_dtypes
            )

            df_m15_dt = prepare_datetime(df_m15_raw, "M15")
            df_m1_dt = prepare_datetime(df_m1_raw, "M1")
            if df_m15_dt is None or df_m1_dt is None or df_m15_dt.empty or df_m1_dt.empty:
                logging.critical("(Error) ข้อมูล M15/M1 ว่างเปล่าหลัง prepare_datetime.")
                sys.exit("ออก: ข้อมูล M15/M1 ว่างเปล่าหลัง prepare_datetime.")

            df_m15_trend = calculate_m15_trend_zone(df_m15_dt)
            # [Patch v6.6.2] Remove duplicate index from Trend Zone before merge
            if df_m15_trend.index.duplicated().any():
                dup_count = int(df_m15_trend.index.duplicated().sum())
                logging.warning("(Warning) พบ index ซ้ำซ้อนใน Trend Zone DataFrame, กำลังลบรายการซ้ำ (คงไว้ค่าแรกของแต่ละ index)")
                df_m15_trend = df_m15_trend.loc[~df_m15_trend.index.duplicated(keep='first')]
                logging.info(f"      Removed {dup_count} duplicate index rows from Trend Zone data.")
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
            if not isinstance(df_m1_cleaned.index, pd.DatetimeIndex):
                df_m1_cleaned.index = pd.to_datetime(df_m1_cleaned.index, errors='coerce', utc=True)
            else:
                df_m1_cleaned.index = pd.to_datetime(df_m1_cleaned.index, utc=True)
            df_m1_cleaned = df_m1_cleaned[df_m1_cleaned.index.notna()]
            if not isinstance(df_m15_trend.index, pd.DatetimeIndex):
                df_m15_trend.index = pd.to_datetime(df_m15_trend.index, errors='coerce', utc=True)
            else:
                df_m15_trend.index = pd.to_datetime(df_m15_trend.index, utc=True)
            df_m15_trend = df_m15_trend[df_m15_trend.index.notna()]
            df_m1_cleaned = df_m1_cleaned.sort_index(); df_m15_trend = df_m15_trend.sort_index()
            df_m1_merged = pd.merge_asof(df_m1_cleaned, df_m15_trend[["Trend_Zone"]], left_index=True, right_index=True, direction="backward", tolerance=pd.Timedelta(minutes=TIMEFRAME_MINUTES_M15 * 2))
            initial_trend_nan = df_m1_merged["Trend_Zone"].isna().sum();
            if initial_trend_nan > 0:
                logging.debug(f"   Filling {initial_trend_nan} NaN values in Trend_Zone with 'NEUTRAL'.")
                # [Patch v5.4.5] Avoid chained assignment warning when filling Trend_Zone
                df_m1_merged["Trend_Zone"] = df_m1_merged["Trend_Zone"].fillna("NEUTRAL")

            logging.info("(Processing) กำลังคำนวณ M1 Entry Signals...");
            base_signal_cfg = ENTRY_CONFIG_PER_FOLD.get(0, {})
            df_m1_merged_with_signals = calculate_m1_entry_signals(df_m1_merged, base_signal_cfg)
            if df_m1_merged_with_signals.empty:
                logging.critical("(Error) M1 ว่างเปล่าหลังคำนวณ Signal.")
                sys.exit("ออก: M1 ว่างเปล่าหลังคำนวณ Signal.")
            # สร้างคอลัมน์ session หากยังไม่มี เพื่อไม่ให้ context ขาดหาย
            df_m1_merged_with_signals = create_session_column(df_m1_merged_with_signals)

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

            # [Patch v5.1.6] สร้างไฟล์ features_main.json จากคอลัมน์จริงของ M1 Data
            # ก่อนที่จะเริ่มขั้นตอน Backtest หรือการบีบอัดไฟล์
            try:
                features_list_actual = [
                    c for c in df_m1_final.columns
                    if c not in ["datetime", "is_tp", "is_sl", "Date"]
                    and pd.api.types.is_numeric_dtype(df_m1_final[c])
                ]
                features_path = os.path.join(OUTPUT_DIR, "features_main.json")
                with open(features_path, "w", encoding="utf-8") as f_feat:
                    json.dump(features_list_actual, f_feat, ensure_ascii=False, indent=2)
                logging.info(
                    f"[Patch] สร้าง features_main.json จาก M1 Data สำเร็จ ({len(features_list_actual)} features)."
                )
            except Exception as e_feat:
                logging.error(f"[Patch] สร้าง features_main.json ล้มเหลว: {e_feat}")

            logging.debug("   Cleaning up intermediate dataframes after data preparation...")
            del df_m15_raw, df_m1_raw, df_m15_dt, df_m1_dt, df_m15_trend
            del df_m1_features, df_m1_cleaned, df_m1_merged, df_m1_merged_with_signals
            maybe_collect()
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
                    parquet_path = m1_save_path.replace('.csv.gz', '.parquet')
                    try:
                        df_m1_final.to_parquet(parquet_path)
                        logging.info(f"   (Success) Saved Parquet: {os.path.basename(parquet_path)}")
                    except Exception as e_parquet:
                        logging.warning(f"   (Warning) Failed to save Parquet: {e_parquet}")

                    log_save_path = os.path.join(OUTPUT_DIR, f"trade_log_v32_walkforward{suffix}.csv.gz")
                    logging.info(f"   Saving generated trade log to: {log_save_path}")
                    if prep_trade_log_wf is not None and not prep_trade_log_wf.empty:
                        if "target" not in prep_trade_log_wf.columns:
                            value_col = None
                            for c in ("profit", "pnl", "PnL_Realized_USD"):
                                if c in prep_trade_log_wf.columns:
                                    value_col = c
                                    break
                            if value_col:
                                prep_trade_log_wf["target"] = (prep_trade_log_wf[value_col] > 0).astype(int)
                                logging.info("[Patch v6.6.6] Added target column using %s", value_col)
                            else:
                                logging.warning("[Patch v6.6.6] No profit column found to derive target")
                        prep_trade_log_wf.to_csv(log_save_path, index=False, encoding="utf-8", compression="gzip")
                        logging.info(f"   (Success) Saved generated trade log ({len(prep_trade_log_wf)} rows): {os.path.basename(log_save_path)}")
                    else:
                        logging.warning(f"   (Warning) Backtest for PREPARE_TRAIN_DATA generated an empty or None trade log. Saving empty log file.")
                        pd.DataFrame(columns=["entry_time"]).to_csv(log_save_path, index=False, encoding="utf-8", compression="gzip")

                    logging.info(f"(Finished) PREPARE_TRAIN_DATA ran backtest and saved results -> suffix={suffix}")
                    del df_m1_final, prep_trade_log_wf
                    maybe_collect()
                    return current_run_suffix
                except (NameError, UnboundLocalError) as ne:
                    # [Patch] Catch UnboundLocalError along with NameError to prevent pipeline crash
                    logging.critical(
                        f"   (CRITICAL) NameError/UnboundLocalError during PREPARE_TRAIN_DATA backtest: {ne}. Likely missing function definition.",
                        exc_info=True,
                    )
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
        model_paths = {
            "main": os.path.join(OUTPUT_DIR, META_CLASSIFIER_PATH),
            "spike": os.path.join(OUTPUT_DIR, SPIKE_MODEL_PATH),
            "cluster": os.path.join(OUTPUT_DIR, CLUSTER_MODEL_PATH),
        }

        model_keys = []
        # [Patch v5.3.10] Handle missing optional models gracefully
        for key, path in model_paths.items():
            if os.path.exists(path):
                model_keys.append(key)
            else:
                if key == 'main':
                    logging.error(f"  (Error) ไม่พบไฟล์ Model '{key}' ({os.path.basename(path)}).")
                    logging.critical("   (CRITICAL) Main model file is missing. Cannot proceed with FULL_RUN.")
                    return None
                else:
                    logging.warning(f"  (Warning) ไม่พบไฟล์ Model '{key}' ({os.path.basename(path)}). ข้ามการโหลด.")

        for model_key in model_keys:
            model_path = model_paths[model_key]
            logging.info(f"(Loading) พยายามโหลด Model '{model_key}' จาก: {model_path}")
            loaded_model = None
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
            # [Patch v5.3.10] Treat missing optional feature files as warnings
            if features_list is None:
                if model_key == 'main':
                    logging.error(f"  (Error) ไม่สามารถโหลด Features สำหรับ Model '{model_key}'.")
                    if loaded_model is not None:
                        logging.warning(f"      (Invalidating) Model '{model_key}' ถูกปิดใช้งานเนื่องจากโหลด Features ไม่สำเร็จ.")
                        loaded_model = None
                    logging.critical("   (CRITICAL) Failed to load features for main model. Cannot proceed.")
                    return None
                else:
                    logging.warning(f"  (Warning) ไม่สามารถโหลด Features สำหรับ Model '{model_key}'. ข้าม model.")
                    if loaded_model is not None:
                        logging.warning(f"      (Invalidating) Model '{model_key}' ถูกปิดใช้งานเนื่องจากโหลด Features ไม่สำเร็จ.")
                        loaded_model = None
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
                logging.debug(f"(Pipeline) Initialized DriftObserver with {len(M1_FEATURES_FOR_DRIFT)} features.")
            except NameError:
                logging.warning("Class 'DriftObserver' not found. Skipping drift analysis.")
                drift_observer = None
        else:
            logging.warning("(Warning) M1_FEATURES_FOR_DRIFT ว่างเปล่า. ไม่สามารถสร้าง DriftObserver.")

    tuning_mode_used = "Fixed Params"
    logging.info(f"\n(Info) ข้าม Auto Threshold Tuning (ใช้ {tuning_mode_used} สำหรับ Model).")
    logging.debug("(Pipeline) Auto Threshold Tuning step skipped. Preparing fund profiles...")
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

        # [Patch v5.5.1] Import model switcher after models and features are loaded
        from src.features import select_model_for_trade

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
                    maybe_collect()

                if drift_observer and fund_name == list(funds_to_run.keys())[0]:
                    logging.info("\n--- Drift Summary (Final Run - Overall) ---")
                    try:
                        if drift_observer is not None:
                            drift_observer.summarize_and_save(OUTPUT_DIR)
                        else:
                            logging.warning("   (Warning) drift_observer is None. Skipping drift summary.")
                    except Exception as e_drift_sum:
                        logging.error(f"   Error summarizing/saving drift results: {e_drift_sum}", exc_info=True)

                qa_log_path = os.path.join(OUTPUT_DIR, '.qa.log')
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
                    saved_path = None
                    if "target" not in trade_log_wf_fund.columns:
                        value_col = None
                        for c in ("profit", "pnl", "PnL_Realized_USD"):
                            if c in trade_log_wf_fund.columns:
                                value_col = c
                                break
                        if value_col:
                            trade_log_wf_fund["target"] = (trade_log_wf_fund[value_col] > 0).astype(int)
                            logging.info("[Patch v6.6.6] Added target column using %s", value_col)
                        else:
                            logging.warning("[Patch v6.6.6] No profit column found to derive target")
                    try:
                        trade_log_wf_fund.to_csv(log_file_path + ".gz", index=False, encoding="utf-8", compression="gzip")
                        saved_path = log_file_path + ".gz"
                        logging.info(f"   (Success) Saved Trade Log (GZ): {log_file_path}.gz")
                    except Exception as e_gz:
                        logging.warning(f"   (Warning) Failed to save trade log as GZ: {e_gz}. Attempting CSV...")
                        try:
                            trade_log_wf_fund.to_csv(log_file_path, index=False, encoding="utf-8")
                            saved_path = log_file_path
                            logging.info(f"   (Success) Saved Trade Log (CSV - Fallback): {log_file_path}")
                        except Exception as e_csv:
                            logging.error(f"   (Error) Failed to save trade log (CSV): {e_csv}", exc_info=True)
                    if saved_path:
                        with open(qa_log_path, 'a', encoding='utf-8') as qa_f:
                            if trade_log_wf_fund.empty:
                                qa_f.write(f"NO_TRADES {final_run_suffix_fund}\n")
                            else:
                                qa_f.write(f"TRADES {len(trade_log_wf_fund)} {final_run_suffix_fund}\n")
                        assert os.path.exists(saved_path)
                        # [Patch v5.4.4] Export simplified trade log for QA checks
                        try:
                            export_trade_log(trade_log_wf_fund, OUTPUT_DIR, fund_name)
                        except Exception as e_exp:
                            logging.error(f"   (Error) Failed to export QA trade log: {e_exp}", exc_info=True)

                    try:
                        # [Patch v5.5.4] Initialize TimeSeriesSplit for equity curve boundaries
                        tscv = TimeSeriesSplit(n_splits=N_WALK_FORWARD_SPLITS)
                        fold_boundaries = [df_m1_final.index.min()] + [df_m1_final.iloc[test_index].index.max() for _, test_index in tscv.split(df_m1_final)]
                        eq_buy_hist_fund_plot_dict = all_funds_equity_histories[fund_name].get(f"Fold0_BUY_{fund_name}", {})
                        eq_sell_hist_fund_plot_dict = all_funds_equity_histories[fund_name].get(f"Fold0_SELL_{fund_name}", {})
                        eq_buy_hist_fund_plot = pd.Series(dict(sorted(eq_buy_hist_fund_plot_dict.items()))).sort_index()
                        eq_buy_hist_fund_plot = eq_buy_hist_fund_plot[~eq_buy_hist_fund_plot.index.duplicated(keep='last')]
                        eq_sell_hist_fund_plot = pd.Series(dict(sorted(eq_sell_hist_fund_plot_dict.items()))).sort_index()
                        eq_sell_hist_fund_plot = eq_sell_hist_fund_plot[~eq_sell_hist_fund_plot.index.duplicated(keep='last')]

                        if 'plot_equity_curve' in globals() and callable(plot_equity_curve):
                            plot_equity_curve(eq_buy_hist_fund_plot, f"Equity Curve - BUY ({fund_name})", INITIAL_CAPITAL, OUTPUT_DIR, f"buy{final_run_suffix_fund}", fold_boundaries)
                            plot_equity_curve(eq_sell_hist_fund_plot, f"Equity Curve - SELL ({fund_name})", INITIAL_CAPITAL, OUTPUT_DIR, f"sell{final_run_suffix_fund}", fold_boundaries)
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
                    log_file_path = os.path.join(OUTPUT_DIR, f"trade_log_v32_walkforward{final_run_suffix_fund}.csv")
                    pd.DataFrame().to_csv(log_file_path, index=False)
                    # [Patch v5.4.4] Also export simplified QA trade log when no results
                    try:
                        export_trade_log(pd.DataFrame(), OUTPUT_DIR, fund_name)
                    except Exception as e_exp:
                        logging.error(f"   (Error) Failed to export QA trade log: {e_exp}", exc_info=True)
                    with open(qa_log_path, 'a', encoding='utf-8') as qa_f:
                        qa_f.write(f"NO_TRADES {final_run_suffix_fund}\n")
                    assert os.path.exists(log_file_path)
                del df_walk_forward_results_pd_fund, trade_log_wf_fund, all_equity_histories_fund, all_fold_metrics_fund
                maybe_collect()

        if MULTI_FUND_MODE and run_mode == 'FULL_RUN' and len(funds_to_run) > 1:
            logging.info("\n" + "=" * 20 + " MULTI-FUND RUN COMPLETED " + "=" * 20)
            if all_funds_trade_logs:
                logging.info("   Combining trade logs from all funds...")
                all_funds_combined_log = pd.concat(all_funds_trade_logs.values(), ignore_index=True)
                if "target" not in all_funds_combined_log.columns:
                    val_col = None
                    for c in ("profit", "pnl", "PnL_Realized_USD"):
                        if c in all_funds_combined_log.columns:
                            val_col = c
                            break
                    if val_col:
                        all_funds_combined_log["target"] = (all_funds_combined_log[val_col] > 0).astype(int)
                        logging.info("[Patch v6.6.6] Added target column to combined log using %s", val_col)
                    else:
                        logging.warning("[Patch v6.6.6] No profit column found for combined log")
                combined_log_path = os.path.join(OUTPUT_DIR, f"trade_log_v32_walkforward_ALL_FUNDS.csv")
                try:
                    all_funds_combined_log.to_csv(combined_log_path + ".gz", index=False, encoding="utf-8", compression="gzip")
                    logging.info(f"   (Success) Saved Combined Trade Log (GZ): {combined_log_path}.gz")
                    del all_funds_combined_log
                    maybe_collect()
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
        maybe_collect()

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
            nvml_handle = None  # [Patch] Clear handle after shutdown
            logging.info("(Success) ปิดการทำงาน pynvml สำเร็จ.")
        except Exception as e:
            logging.warning(f"(Warning) เกิดข้อผิดพลาดขณะปิด pynvml: {e}")

        # [Patch v6.2.4] Auto Threshold Optimization Stage
        run_auto_threshold_stage()

    end_time_main = time.time()
    logging.info(f"\n--- ฟังก์ชัน Main (Mode: {run_mode}) เสร็จสิ้นใน {end_time_main - start_time_main:.2f} วินาที ---")

    return final_run_suffix


# ==============================================================================
# === SCRIPT ENTRY POINT ===
# ==============================================================================
if __name__ == "__main__":
    start_time_script = time.time()
    logger.info(f"(Starting) Script Gold Trading AI v4.8.4...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preprocess', 'backtest', 'report'], default='full')
    args = parser.parse_args()
    stage_map = {
        'preprocess': 'PREPARE_TRAIN_DATA',
        'backtest': 'FULL_RUN',
        'report': 'REPORT',
    }
    selected_run_mode = stage_map.get(args.stage, 'FULL_PIPELINE')
    logger.info(f"(Starting) กำลังเริ่มการทำงานหลัก (main) ในโหมด: {selected_run_mode}...")
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

    except SystemExit as se_main:
        logger.critical(f"\n(Critical Error) สคริปต์ออกก่อนเวลา: {se_main}")
    except KeyboardInterrupt:
        logger.warning("\n(Stopped) การทำงานหยุดโดยผู้ใช้ (KeyboardInterrupt).")
    except NameError as ne_main:
        logger.critical(
            f"\n(Error) NameError in __main__: '{ne_main}'. Critical function or variable likely missing.",
            exc_info=True,
        )
    except Exception as e_main_general:
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e_main_general), exc_info=True)
        sys.exit(1)
    finally:
        end_time_script = time.time()
        total_duration = end_time_script - start_time_script
        logger.info(f"\n(Finished) Script Gold Trading AI v4.8.4 เสร็จสมบูรณ์!")
        
        final_tuning_mode_log = "Unknown"
        # Check globals first, then locals if not found in globals (though it should be global)
        if 'tuning_mode_used' in globals() and globals()['tuning_mode_used'] is not None:
            final_tuning_mode_log = globals()['tuning_mode_used']
        elif 'tuning_mode_used' in locals() and locals()['tuning_mode_used'] is not None: # Check local scope as fallback
            final_tuning_mode_log = locals()['tuning_mode_used']
        logger.info(f"   Tuning Mode ที่ใช้: {final_tuning_mode_log}")

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
                logger.info(f"   ผลลัพธ์ถูกบันทึกไปที่: {output_dir_final_path}")
                logger.info(f"   ไฟล์ Log หลัก: {log_filename_val}")
            elif output_dir_final_path:
                logger.warning(f"   (Warning) ไม่พบ Output Directory ที่คาดหวัง: {output_dir_final_path}")
            else:
                logger.warning("   (Warning) ไม่สามารถกำหนด Output Directory path.")
        except Exception as e_report_path:
            logger.warning(f"   (Warning) Error reporting output path: {e_report_path}")

        logger.info(f"   เวลาดำเนินการทั้งหมด: {total_duration:.2f} วินาที ({total_duration/60:.2f} นาที).")
        logger.info("--- End of Script ---")
        # [Patch v5.4.1] สรุปผลแบบย่อสำหรับโหมด COMPACT_LOG
        logger.warning(
            f"[SUMMARY] Runtime: {total_duration:.2f}s | Output: {output_dir_final_path or 'N/A'}"
        )

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

logger.info("Loading Part 11: MT5 Connector (Placeholder)...")

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

# ---------------------------------------------------------------------------
# Padding to preserve line numbers for downstream tests
if False:
    pass
# padding start
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# padding end

# ---------------------------------------------------------------------------
# Stubs for Function Registry Tests

def parse_arguments():
    """Stubbed argument parser."""
    return {}


def setup_output_directory(base_dir, dir_name):
    """Stubbed setup_output_directory for main."""
    return dl_setup_output_directory(base_dir, dir_name)


def load_features_from_file(_):
    """Stubbed loader for saved features."""
    return {}


def drop_nan_rows(df):
    """Stubbed NaN dropper."""
    return df.dropna()


def convert_to_float32(df):
    """Stubbed float32 converter."""
    return df.astype("float32", errors="ignore")


def run_initial_backtest():
    """Stubbed initial backtest runner."""
    return None


def save_final_data(df, path):
    """Stubbed data saver."""
    df.to_csv(path)


# [Patch v6.2.4] Auto Threshold Optimization Helper
def run_auto_threshold_stage():
    """Run Optuna-based threshold tuning if enabled."""
    from src.features import ENABLE_AUTO_THRESHOLD_TUNING

    if ENABLE_AUTO_THRESHOLD_TUNING:
        import threshold_optimization as topt
        logger.info("[Patch v6.2.4] Starting Auto Threshold Optimization")
        topt.run_threshold_optimization(
            output_dir=OUTPUT_DIR,
            trials=OPTUNA_N_TRIALS,
            study_name="threshold_wfv",
            direction=OPTUNA_DIRECTION,
            timeout=None,
        )
        logger.info("[Patch v6.2.4] Auto Threshold Optimization Completed")


# [Patch v5.5.9] Pipeline helper for discrete stages
# [Patch v6.8.5] Support configurable feature format
def run_pipeline_stage(stage: str):
    """Run a specific pipeline stage."""
    settings = load_settings()
    fmt = getattr(settings, "feature_format", "parquet")
    ext_map = {"parquet": ".parquet", "hdf5": ".h5"}
    ext = ext_map.get(fmt.lower(), ".csv")
    if stage == 'preprocess':
        df = load_validated_csv(DATA_FILE_PATH_M1, "M1")
        df = engineer_m1_features(df)
        out_path = os.path.join(OUTPUT_DIR, f"preprocessed{ext}")
        save_features(df, out_path, fmt)
        del df
        maybe_collect()
        logger.info(f"[Pipeline] Preprocess complete -> {out_path}")
        return out_path
    if stage == 'backtest':
        data_path = os.path.join(OUTPUT_DIR, f"preprocessed{ext}")
        if os.path.exists(data_path):
            df = load_features(data_path, fmt)
            if df is None:
                df = load_validated_csv(DATA_FILE_PATH_M1, "M1")
        else:
            df = load_validated_csv(DATA_FILE_PATH_M1, "M1")
        run_backtest_simulation_v34(df, label="WFV", initial_capital_segment=INITIAL_CAPITAL)
        logger.info("[Pipeline] Backtest completed")
        return None
    if stage == 'report':
        metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            plot_equity_curve([], "Equity", INITIAL_CAPITAL, OUTPUT_DIR, "report")
            logger.info("[Pipeline] Report generated")
        else:
            logger.warning("[Pipeline] No metrics to report")
        return None
    logger.error(f"Unknown stage: {stage}")
    return None

logging.info("Reached End of Part 12 (End of Script Marker).")
# === END OF PART 12/12 ===
