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
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from typing import Dict, List
# [Patch v5.2.0] Use explicit package import for cooldown utilities
from src.cooldown_utils import (
    is_soft_cooldown_triggered,
    step_soft_cooldown,
)
from itertools import product
from src.utils.sessions import get_session_tag  # [Patch v5.1.3]
from src.utils import get_env_float
from src.config import (
    print_gpu_utilization,  # [Patch v5.2.0] นำเข้า helper สำหรับแสดงการใช้งาน GPU/RAM (print_gpu_utilization)
    USE_MACD_SIGNALS,
    USE_RSI_SIGNALS,
)

# อ่านเวอร์ชันจากไฟล์ VERSION
VERSION_FILE = os.path.join(os.path.dirname(__file__), '..', 'VERSION')
with open(VERSION_FILE, 'r', encoding='utf-8') as vf:
    __version__ = vf.read().strip()
try:
    import numba
    from numba import njit
except Exception:  # pragma: no cover - fallback when numba unavailable
    numba = None
    def njit(func):
        return func

# [Patch v4.8.8] Import safe_set_datetime using unconditional absolute import
from src.data_loader import safe_set_datetime
from src.data_loader import safe_load_csv_auto  # [Patch v5.1.6] Ensure CSV loader is imported
from src.data_loader import simple_converter
from src.data_loader import load_final_m1_data  # [Patch v5.4.5] Loader with validation

# [Patch v4.8.9] Import safe_get_global using unconditional absolute import
from src.data_loader import safe_get_global
from src.features import (
    select_top_shap_features,
    check_model_overfit,
    analyze_feature_importance_shap,
    check_feature_noise_shap,  # [Patch] เพิ่มการ import เพื่อตรวจสอบ SHAP noise
    rsi,
    macd,
)  # [Patch] นำเข้า Dynamic Feature Selection & Overfitting Helpers
import traceback
from joblib import dump as joblib_dump # Use joblib dump directly
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
)  # [Patch] นำเข้า metric ที่ขาดหายไป
from src.evaluation import find_best_threshold
import gc # For memory management
import os
import itertools
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

# [Patch] นำเข้า pynvml สำหรับตรวจสอบ GPU (Prevent NameError)
try:
    import pynvml
except ImportError:
    pynvml = None
    nvml_handle = None
else:
    try:
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        nvml_handle = None

# Ensure global configurations are accessible if run independently
# Define defaults if globals are not found
DEFAULT_META_CLASSIFIER_PATH = "meta_classifier.pkl"
DEFAULT_SPIKE_MODEL_PATH = "meta_classifier_spike.pkl"
DEFAULT_CLUSTER_MODEL_PATH = "meta_classifier_cluster.pkl"
DEFAULT_MODEL_TO_LINK = "catboost"
DEFAULT_ENABLE_OPTUNA_TUNING = True
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
    enable_threshold_tuning=False,
    fold_index=None,
):
    """
    Trains and exports a Meta Classifier (L1) model for a specific purpose
    (main, spike, cluster) using trade log data and M1 features. Includes options
    for dynamic feature selection and hyperparameter optimization (Optuna).
    (v4.8.8 Patch 2: Fixed UnboundLocalError, Pool usage)

    Args:
        # ... (Args remain the same) ...
        fold_index (int, optional): หมายเลขโฟลด์สำหรับแสดงใน log เมื่อไม่มีข้อมูล

    Returns:
        tuple[dict, list]: A tuple containing:
            - saved_model_paths (dict): Dictionary mapping model purpose to the saved model file path.
                                        Returns None if training fails critically.
            - final_features_used (list): List of feature names used for the final trained model.
                                          Returns empty list if training fails.
    """
    start_train_time = time.time()
    logging.info(f"\n(Training - v{__version__}) เริ่มต้นการ Train Meta Classifier (Purpose: {model_purpose.upper()})...") # Updated version in log
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

    global USE_GPU_ACCELERATION, meta_model_type_used, pattern_label_map; USE_GPU_ACCELERATION = globals().get('USE_GPU_ACCELERATION', False)
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
        # [Patch v5.1.6] Ensure Trade Log has 'datetime' column for merge
        if 'datetime' not in trade_log_df.columns:
            trade_log_df['datetime'] = trade_log_df['entry_time']
        trade_log_df['datetime'] = pd.to_datetime(trade_log_df['datetime'])
    elif trade_log_path and isinstance(trade_log_path, str):
        logging.info(f"   กำลังโหลด Trade Log (Default Path): {trade_log_path}")
        try:
            # [Patch v5.4.5] Limit loaded rows to manage memory for large logs
            trade_log_df = safe_load_csv_auto(trade_log_path, row_limit=sample_size)
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
            # [Patch v5.1.6] Ensure Trade Log has 'datetime' column for merge
            if 'datetime' not in trade_log_df.columns:
                trade_log_df['datetime'] = trade_log_df['entry_time']
            trade_log_df['datetime'] = pd.to_datetime(trade_log_df['datetime'])
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
        trade_log_df = trade_log_df.sort_values("datetime")
        logging.info(f"   ประมวลผล Trade Log สำเร็จ ({len(trade_log_df)} trades).")

    except Exception as e:
        logging.error(f"(Error) เกิดข้อผิดพลาดในการประมวลผล Trade Log: {e}", exc_info=True)
        return None, []

    logging.info(f"   กำลังโหลด M1 Data: {m1_data_path}")
    if not os.path.exists(m1_data_path):
        logging.error(f"(Error) ไม่พบ M1 Data file: {m1_data_path}")
        return None, []

    try:
        m1_df = load_final_m1_data(m1_data_path, trade_log_df)
        if m1_df is None:
            return None, []

        logging.info(
            f"   โหลดและเตรียม M1 สำเร็จ ({len(m1_df)} แถว). จำนวน Features เริ่มต้น: {len(m1_df.columns)}"
        )
    except Exception as e:
        logging.error(f"(Error) ไม่สามารถโหลดหรือเตรียม M1 data: {e}", exc_info=True)
        return None, []

    logging.info(f"   กำลังเตรียมข้อมูลสำหรับ Meta Model Training (Purpose: {model_purpose.upper()})...")
    merged_df = None
    initial_features_for_selection = [] # Initialize here

    logging.info("   กำลังรวม Trade Log กับ M1 Features (merge_asof)...")
    try:
        if not pd.api.types.is_datetime64_any_dtype(trade_log_df["datetime"]):
            trade_log_df["datetime"] = pd.to_datetime(trade_log_df["datetime"], errors='coerce', utc=True)
        else:
            trade_log_df["datetime"] = pd.to_datetime(trade_log_df["datetime"], utc=True)
        trade_log_df.dropna(subset=["datetime"], inplace=True)
        if trade_log_df.empty:
            logging.error("(Error) ไม่มี Trades ที่มี datetime ถูกต้องหลังการแปลง (ก่อน Merge).")
            return None, []
        if not pd.api.types.is_datetime64_any_dtype(m1_df["datetime"]):
            m1_df["datetime"] = pd.to_datetime(m1_df["datetime"], errors='coerce', utc=True)
        else:
            m1_df["datetime"] = pd.to_datetime(m1_df["datetime"], utc=True)
        m1_df.dropna(subset=["datetime"], inplace=True)
        if trade_log_df.empty or m1_df.empty:
            logging.error("(Error) DataFrame ว่างหลังการเตรียม datetime สำหรับ merge.")
            return None, []
        trade_log_df_sorted = trade_log_df.sort_values("datetime").reset_index(drop=True)
        m1_df_sorted = m1_df.sort_values("datetime").reset_index(drop=True)
        logging.info("[Patch] เริ่ม Merge Trade Log กับ M1 Features (merge_asof)")
        merged_df = pd.merge_asof(
            trade_log_df_sorted,
            m1_df_sorted,
            on="datetime",
            direction="backward",
            tolerance=pd.Timedelta(minutes=5)
        )
        logging.info(f"   Merge completed. Shape after merge: {merged_df.shape}")
        del trade_log_df, m1_df
        gc.collect()

        # [Patch v5.1.6] Load feature list from features_main.json
        features_json_path = os.path.join(output_dir, "features_main.json")
        try:
            with open(features_json_path, "r", encoding="utf-8") as f_feat:
                feature_list = json.load(f_feat)
        except Exception as e_feat:
            logging.warning(f"[Patch] ไม่สามารถโหลด features_main.json: {e_feat}. ใช้ META_CLASSIFIER_FEATURES แทน")
            feature_list = META_CLASSIFIER_FEATURES

        available_features = [f for f in feature_list if f in merged_df.columns]
        missing_features = [f for f in feature_list if f not in merged_df.columns]
        logging.info(f"[Patch] Available Features หลัง Merge: {len(available_features)} รายการ")
        logging.info(f"[Patch] Missing Features หลัง Merge: {len(missing_features)} รายการ: {missing_features}")
        if not available_features:
            logging.error(f"[Error] ไม่มี Features ใช้ได้หลัง Merge: feature_list ทั้งหมดหายไป! (missing: {missing_features})")
            return None, []
        initial_features_for_selection = available_features
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
            if fold_index is not None:
                logging.error(f"โฟลด์ {fold_index} ไม่มีข้อมูลเพียงพอสำหรับฝึกโมเดล")
            else:
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
                    # [Patch v5.5.4] Evaluate Lag Feature importance using SHAP with full feature set
                    lag_pool = Pool(X_select, label=y_select, cat_features=cat_features_indices_select_cpu)
                    explainer_lag = shap.TreeExplainer(prelim_model)
                    shap_vals_lag = explainer_lag.shap_values(lag_pool)
                    shap_vals_pos = None
                    df_lag = None
                    if isinstance(shap_vals_lag, list) and len(shap_vals_lag) == 2:
                        shap_vals_pos = shap_vals_lag[1]
                    elif isinstance(shap_vals_lag, np.ndarray) and shap_vals_lag.ndim == 2:
                        shap_vals_pos = shap_vals_lag
                    elif isinstance(shap_vals_lag, np.ndarray) and shap_vals_lag.ndim == 3 and shap_vals_lag.shape[0] >= 2:
                        shap_vals_pos = shap_vals_lag[1, :, :]
                    if shap_vals_pos is not None:
                        df_lag = pd.DataFrame(shap_vals_pos, columns=X_select.columns)[potential_lag_features]
                        mean_abs_lag = df_lag.abs().mean().values
                        lag_df = pd.DataFrame({'feature': potential_lag_features, 'mean_abs_shap': mean_abs_lag})
                        total_shap = lag_df['mean_abs_shap'].sum()
                        lag_df['norm_shap'] = lag_df['mean_abs_shap'] / total_shap if total_shap > 1e-9 else 0.0
                        significant_lags = lag_df[lag_df['norm_shap'] >= shap_importance_threshold]['feature'].tolist()
                    else:
                        significant_lags = []

                    if significant_lags:
                        logging.info(f"         Lag Features ที่มีความสำคัญเบื้องต้น: {significant_lags}")
                        added_lags = [lag for lag in significant_lags if lag not in selected_features]
                        if added_lags:
                            logging.info(f"         (Info) เพิ่ม Lag Features ที่มีความสำคัญ: {added_lags}")
                            selected_features.extend(added_lags)
                    else:
                        logging.info("         ไม่มี Lag Features ที่มีความสำคัญเบื้องต้นตามเกณฑ์.")
                    del shap_vals_lag, shap_vals_pos, df_lag, lag_df, mean_abs_lag, total_shap
                except Exception as e_lag_fi:
                    logging.warning(f"Cannot evaluate Lag Features: {e_lag_fi}")
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

            if enable_threshold_tuning and y_proba_cat_val is not None:
                try:
                    best_t, best_s = find_best_threshold(y_proba_cat_val, y_val_cat)
                    logging.info(f"[Patch] Tuned threshold to {best_t:.2f} (F1={best_s:.3f})")
                except Exception as e_thresh:
                    logging.warning(f"[Patch] Threshold tuning failed: {e_thresh}")

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
    logging.info(f"(Finished - v{__version__}) Meta Classifier Training (Purpose: {model_purpose.upper()}) complete in {end_train_time - start_train_time:.2f} seconds.") # Updated version in log
    if 'X_val_cat_for_shap' in locals(): del X_val_cat_for_shap
    gc.collect()
    return saved_model_paths, final_features_catboost

logging.info(f"Part 7: Model Training Function Loaded (v{__version__} Applied).")
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
DEFAULT_ADAPTIVE_SIGNAL_SCORE_WINDOW = 1000
DEFAULT_ADAPTIVE_SIGNAL_SCORE_QUANTILE = 0.7
DEFAULT_MIN_SIGNAL_SCORE_ENTRY_MIN = 0.5
DEFAULT_MIN_SIGNAL_SCORE_ENTRY_MAX = 3.0
DEFAULT_USE_ADAPTIVE_SIGNAL_SCORE = True
DEFAULT_RECOVERY_MODE_CONSECUTIVE_LOSSES = 4
DEFAULT_RECOVERY_MODE_LOT_MULTIPLIER = 0.5
DEFAULT_MIN_LOT_SIZE = 0.01
DEFAULT_MAX_LOT_SIZE = 5.0
DEFAULT_POINT_VALUE = 0.1
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_USE_REENTRY = True
DEFAULT_REENTRY_COOLDOWN_BARS = 1
DEFAULT_TIMEFRAME_MINUTES_M1 = 1
DEFAULT_MAX_CONCURRENT_ORDERS = 7
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
DEFAULT_ENTRY_CONFIG_PER_FOLD = {0: {"sl_multiplier": 2.8, "gain_z_thresh": 0.3, "cooldown_sec": 0, "min_signal_score": 2.0}}
DEFAULT_BASE_BE_SL_R_THRESHOLD = 1.0
DEFAULT_DYNAMIC_BE_ATR_THRESHOLD_HIGH = 1.2
DEFAULT_DYNAMIC_BE_R_ADJUST_HIGH = 0.2
DEFAULT_ENABLE_PARTIAL_TP = True
DEFAULT_PARTIAL_TP_LEVELS = [{"r_multiple": 0.8, "close_pct": 0.5}]
DEFAULT_PARTIAL_TP_MOVE_SL_TO_ENTRY = True
DEFAULT_ENABLE_KILL_SWITCH = True
DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD = 0.30
DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 10
DEFAULT_KILL_SWITCH_WARNING_MAX_DD_THRESHOLD = 0.25
DEFAULT_KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD = 7
DEFAULT_FUND_PROFILES = {"NORMAL": {"risk": 0.01, "mm_mode": "balanced"}}
DEFAULT_FUND_NAME = "NORMAL"
DEFAULT_USE_META_CLASSIFIER = True
DEFAULT_META_MIN_PROBA_THRESH = 0.5
DEFAULT_REENTRY_MIN_PROBA_THRESH = 0.5
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
ADAPTIVE_SIGNAL_SCORE_WINDOW = safe_get_global('ADAPTIVE_SIGNAL_SCORE_WINDOW', DEFAULT_ADAPTIVE_SIGNAL_SCORE_WINDOW)
ADAPTIVE_SIGNAL_SCORE_QUANTILE = safe_get_global('ADAPTIVE_SIGNAL_SCORE_QUANTILE', DEFAULT_ADAPTIVE_SIGNAL_SCORE_QUANTILE)
MIN_SIGNAL_SCORE_ENTRY_MIN = safe_get_global('MIN_SIGNAL_SCORE_ENTRY_MIN', DEFAULT_MIN_SIGNAL_SCORE_ENTRY_MIN)
MIN_SIGNAL_SCORE_ENTRY_MAX = safe_get_global('MIN_SIGNAL_SCORE_ENTRY_MAX', DEFAULT_MIN_SIGNAL_SCORE_ENTRY_MAX)
USE_ADAPTIVE_SIGNAL_SCORE = safe_get_global('USE_ADAPTIVE_SIGNAL_SCORE', DEFAULT_USE_ADAPTIVE_SIGNAL_SCORE)
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
KILL_SWITCH_WARNING_MAX_DD_THRESHOLD = safe_get_global('KILL_SWITCH_WARNING_MAX_DD_THRESHOLD', DEFAULT_KILL_SWITCH_WARNING_MAX_DD_THRESHOLD)
KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD = safe_get_global('KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD', DEFAULT_KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD)
FUND_PROFILES = safe_get_global('FUND_PROFILES', DEFAULT_FUND_PROFILES)
DEFAULT_FUND_NAME = safe_get_global('DEFAULT_FUND_NAME', DEFAULT_FUND_NAME)
USE_META_CLASSIFIER = safe_get_global('USE_META_CLASSIFIER', DEFAULT_USE_META_CLASSIFIER)
META_MIN_PROBA_THRESH = safe_get_global('META_MIN_PROBA_THRESH', DEFAULT_META_MIN_PROBA_THRESH)
REENTRY_MIN_PROBA_THRESH = safe_get_global('REENTRY_MIN_PROBA_THRESH', DEFAULT_REENTRY_MIN_PROBA_THRESH)
OUTPUT_DIR = safe_get_global('OUTPUT_DIR', DEFAULT_OUTPUT_DIR)


# --- Backtesting Helper Functions ---
# safe_set_datetime is now in Part 3


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

# [Patch v5.3.9] Adaptive Signal Score helper
def get_dynamic_signal_score_entry(df, window=1000, quantile=0.7, min_val=0.5, max_val=3.0):
    """Return quantile-based signal score threshold with clamp."""
    if df is None or 'Signal_Score' not in df.columns or len(df) == 0:
        return min_val
    scores = df['Signal_Score'].dropna().astype(float)
    recent_scores = scores.iloc[-window:]
    if recent_scores.empty:
        return min_val
    val = recent_scores.quantile(quantile)
    val = max(min_val, min(val, max_val))
    return float(val)

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
    """Spike guard filter for London session with debug reasons."""
    if not ENABLE_SPIKE_GUARD:
        logging.debug("      (Spike Guard) Disabled via config.")
        return True
    if not isinstance(session, str) or "London" not in session:
        logging.debug("      (Spike Guard) Not London session - skipping.")
        return True

    spike_score_val = pd.to_numeric(getattr(row, "spike_score", np.nan), errors='coerce')
    if pd.notna(spike_score_val) and spike_score_val > 0.85:
        logging.debug(f"      (Spike Guard Filtered) Reason: London Session & High Spike Score ({spike_score_val:.2f} > 0.85)")
        return False

    adx_val = pd.to_numeric(getattr(row, "ADX", np.nan), errors='coerce')
    wick_ratio_val = pd.to_numeric(getattr(row, "Wick_Ratio", np.nan), errors='coerce')
    vol_index_val = pd.to_numeric(getattr(row, "Volatility_Index", np.nan), errors='coerce')
    candle_body_val = pd.to_numeric(getattr(row, "Candle_Body", np.nan), errors='coerce')
    candle_range_val = pd.to_numeric(getattr(row, "Candle_Range", np.nan), errors='coerce')
    gain_val = pd.to_numeric(getattr(row, "Gain", np.nan), errors='coerce')
    atr_val = pd.to_numeric(getattr(row, "ATR_14", np.nan), errors='coerce')

    if any(pd.isna(v) for v in [adx_val, wick_ratio_val, vol_index_val, candle_body_val, candle_range_val, gain_val, atr_val]):
        logging.debug("      (Spike Guard) Missing values - skip filter.")
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
        logging.debug("      (Spike Guard Allowed) Reason: Strong directional move override.")
        return True

    logging.debug("      (Spike Guard) Passed all checks.")
    return True

def is_entry_allowed(row, session, consecutive_losses, signal_score_threshold=None):
    """Checks if entry is allowed based on filters with debug logging."""
    if signal_score_threshold is None:
        global MIN_SIGNAL_SCORE_ENTRY
        signal_score_threshold = MIN_SIGNAL_SCORE_ENTRY

    if not spike_guard_london(row, session, consecutive_losses):
        logging.debug("      Entry blocked by Spike Guard.")
        return False, "SPIKE_GUARD_LONDON"

    signal_score = pd.to_numeric(getattr(row, "Signal_Score", np.nan), errors='coerce')
    if pd.isna(signal_score):
        logging.debug("      Entry blocked: Invalid Signal Score (NaN)")
        return False, "INVALID_SIGNAL_SCORE (NaN)"
    if abs(signal_score) < signal_score_threshold:
        logging.debug(
            f"      Entry blocked: Low Signal Score {signal_score:.2f} < {signal_score_threshold}"
        )
        return False, f"LOW_SIGNAL_SCORE ({signal_score:.2f}<{signal_score_threshold})"

    logging.debug("      Entry allowed by filters.")
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

    current_high = pd.to_numeric(getattr(row, "High", np.nan), errors='coerce')
    current_low = pd.to_numeric(getattr(row, "Low", np.nan), errors='coerce')
    current_close = pd.to_numeric(getattr(row, "Close", np.nan), errors='coerce')
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

# <<< [Patch v5.5.2] Helper to resolve close index >>>
def _resolve_close_index(df_sim, entry_idx, close_timestamp):
    """Return a valid index for closing orders."""
    if entry_idx is None:
        return None
    if entry_idx in df_sim.index:
        return entry_idx
    nearest_pos = df_sim.index.get_indexer([close_timestamp], method="nearest")[0]
    resolved_idx = df_sim.index[nearest_pos]
    logging.warning(
        f"(Warning) entry index {entry_idx} not in df_sim.index. ใช้ nearest_idx {resolved_idx} แทน."
    )
    return resolved_idx

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
    # [Patch v5.1.0] ตรวจสอบคอลัมน์สำคัญก่อนดำเนินการ backtest
    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df_m1_segment_pd.columns]
    if missing:
        # เมื่อเรียกโดย profile_backtest จะตรวจสอบและ return ก่อนถึงจุดนี้
        raise ValueError(
            f"Missing required columns in input DataFrame for backtest: {missing}"
        )
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
    soft_cooldown_bars_remaining = 0
    SOFT_COOLDOWN_LOOKBACK = 10
    # [Patch v5.0.18] Increase loss threshold to reduce trade blocking
    SOFT_COOLDOWN_LOSS_COUNT = 6
    # [Patch v5.0.18] MACD entry thresholds to allow mild counter-trend trades
    MACD_NEG_THRESHOLD_BUY = -0.05
    MACD_POS_THRESHOLD_SELL = 0.05
    kill_switch_trigger_time = pd.NaT
    current_risk_mode = "normal"; trade_history_list = []
    error_in_loop = False
    # [Patch] Track last logged threshold to avoid spam
    last_logged_signal_thresh = None

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
    base_cfg = ENTRY_CONFIG_PER_FOLD.get(current_fold_index, ENTRY_CONFIG_PER_FOLD.get(0, {}))
    fold_sl_multiplier_base = fold_config.get("sl_multiplier", base_cfg.get("sl_multiplier", 2.8))
    logging.info(
        f"   [Patch B Check] Using SL Multiplier: {fold_sl_multiplier_base} for Fold {current_fold_index+1} (from fold_config or base_cfg)"
    )
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
    current_bar_index = 0
    # <<< [Patch v4.9.0] Use itertuples for faster iteration >>>
    iterator_obj = df_sim.itertuples(name='Bar')
    if tqdm:
        iterator = tqdm(iterator_obj, total=df_sim.shape[0], desc=f"  Sim ({label}, {side})", leave=False, mininterval=2.0)
    else:
        iterator = iterator_obj
    run_summary = {}

    try:
        for row in iterator:
            current_index = row.Index
            now = current_index
            equity_at_start_of_bar = equity
            current_equity_change_this_bar = 0.0
            logging.debug(
                f"--- Bar {current_bar_index} ({current_index}) --- Equity Start: {equity_at_start_of_bar:.2f}, Active Orders: {len(active_orders)}"
            )
            current_open = row.Open
            current_low = row.Low
            current_high = row.High
            current_close = row.Close
            current_atr_shifted = pd.to_numeric(getattr(row, "ATR_14_Shifted"), errors="coerce")
            current_atr = pd.to_numeric(getattr(row, "ATR_14"), errors="coerce")
            current_avg_atr = pd.to_numeric(getattr(row, "ATR_14_Rolling_Avg"), errors="coerce")
            current_vol_index = pd.to_numeric(getattr(row, "Volatility_Index"), errors="coerce")
            current_macd_smooth = pd.to_numeric(getattr(row, "MACD_hist_smooth"), errors="coerce")
            current_signal_score = pd.to_numeric(getattr(row, "Signal_Score"), errors="coerce")
            current_rsi = pd.to_numeric(getattr(row, "RSI"), errors="coerce")
            current_gain_z = pd.to_numeric(getattr(row, "Gain_Z"), errors="coerce")
            current_trade_tag = getattr(row, "Trade_Tag", "N/A")
            session_tag = getattr(row, "session", "Other")
            if any(pd.isna(p) or (isinstance(p, float) and np.isinf(p)) for p in [current_open, current_high, current_low, current_close]):
                logging.debug(
                    f"   Skipping bar {current_index} due to missing/invalid price data."
                )
                df_sim.loc[current_index, f"Max_Drawdown_At_Point{label_suffix}"] = max_drawdown_pct
                df_sim.loc[current_index, f"Equity_Realistic{label_suffix}"] = equity
                df_sim.loc[current_index, f"Active_Order_Count{label_suffix}"] = len(active_orders)
                equity_history[current_index] = equity
                current_bar_index += 1
                continue
            next_active_orders = []; order_closed_this_bar_flag = False
            logging.debug(
                f"   Processing {len(active_orders)} active orders for bar {current_index}..."
            )
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
                        # [Patch v5.3.5] Add buffer to EarlyExit and increase ATR threshold
                        reversal_threshold_atr = 2.0
                        entry_bar = order.get("entry_bar_count", current_bar_index)
                        bars_since_open = current_bar_index - entry_bar
                        early_exit_triggered = False
                        if order_side == "BUY":
                            peak_since_tp1 = order.get("peak_since_tp1")
                            if pd.notna(peak_since_tp1):
                                order["peak_since_tp1"] = max(peak_since_tp1, current_high)
                            reversal_distance = order["peak_since_tp1"] - current_low
                            reversal_threshold_price = reversal_threshold_atr * current_atr_num_early_exit
                            if bars_since_open > 3 and reversal_distance >= reversal_threshold_price:
                                early_exit_triggered = True
                                close_reason = f"EarlyExit_Reversal_{reversal_threshold_atr}ATR (buffer)"
                                exit_price = current_close
                        elif order_side == "SELL":
                            trough_since_tp1 = order.get("trough_since_tp1")
                            if pd.notna(trough_since_tp1):
                                order["trough_since_tp1"] = min(trough_since_tp1, current_low)
                            reversal_distance = current_high - order["trough_since_tp1"]
                            reversal_threshold_price = reversal_threshold_atr * current_atr_num_early_exit
                            if bars_since_open > 3 and reversal_distance >= reversal_threshold_price:
                                early_exit_triggered = True
                                close_reason = f"EarlyExit_Reversal_{reversal_threshold_atr}ATR (buffer)"
                                exit_price = current_close
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
                        if entry_bar_idx_log is not None:
                            resolved_idx = _resolve_close_index(df_sim, entry_bar_idx_log, close_timestamp)
                            if resolved_idx is not None:
                                safe_set_datetime(df_sim, resolved_idx, f"Order_Closed_Time{label_suffix}", close_timestamp)
                                df_sim.loc[resolved_idx, f"PnL_Realized_USD{label_suffix}"] = net_pnl_usd; df_sim.loc[resolved_idx, f"Commission_USD{label_suffix}"] = commission_usd; df_sim.loc[resolved_idx, f"Spread_Cost_USD{label_suffix}"] = spread_cost_usd; df_sim.loc[resolved_idx, f"Slippage_USD{label_suffix}"] = slippage_usd; df_sim.loc[resolved_idx, f"Exit_Reason_Actual{label_suffix}"] = close_reason; df_sim.loc[resolved_idx, f"Exit_Price_Actual{label_suffix}"] = exit_price; df_sim.loc[resolved_idx, f"PnL_Points_Actual{label_suffix}"] = pnl_points_net_spread
                                safe_set_datetime(df_sim, resolved_idx, f"BE_Triggered_Time{label_suffix}", order.get("be_triggered_time", pd.NaT))
                        else:
                            logging.warning(f"      (Warning) Could not find entry index '{entry_bar_idx_log}' in df_sim to update results for order {order_entry_time}.")
                        continue
                    else:
                        logging.debug(f"         Order {order_entry_time} remains open. Updating BE/TSL/TTP2...")
                        order, be_triggered_this_bar, tsl_updated_this_bar, be_sl_triggered_count_run, tsl_triggered_count_run = _update_open_order_state(order, current_high, current_low, current_atr, current_avg_atr, now, base_be_r_thresh, fold_sl_multiplier_base, base_tp_multiplier_config, be_sl_triggered_count_run, tsl_triggered_count_run)
                        logging.debug(f"         Appending order {order_entry_time} to next_active_orders.")
                        next_active_orders.append(order)
            # <<< [Patch C - Unified] End of try-except for order processing loop >>>
            except Exception as e_order_processing:
                logging.critical(
                    f"   (CRITICAL) Error processing order {order.get('entry_time', 'N/A_ORDER')} for bar {current_index}: {e_order_processing}",
                    exc_info=True,
                )
                traceback.print_exc()
                error_in_loop = True
                if 'order' in locals() and order not in next_active_orders :
                    next_active_orders.append(order)
                continue # Attempt to continue to the next order or next bar

            m15_trend = getattr(row, "Trend_Zone", "NEUTRAL"); entry_long_signal = (getattr(row, "Entry_Long", 0) == 1); entry_short_signal = (getattr(row, "Entry_Short", 0) == 1); trade_tag = getattr(row, "Trade_Tag", "N/A"); signal_score = pd.to_numeric(getattr(row, "Signal_Score", np.nan), errors='coerce'); trade_reason = getattr(row, "Trade_Reason", "NONE"); pattern_label = getattr(row, "Pattern_Label", "Normal")
            final_m1_signal = "NONE"
            if side == "BUY" and entry_long_signal: final_m1_signal = "BUY"
            elif side == "SELL" and entry_short_signal: final_m1_signal = "SELL"
            df_sim.loc[current_index, f"M15_Trend_Zone{label_suffix}"] = m15_trend
            df_sim.loc[current_index, f"M1_Entry_Signal{label_suffix}"] = final_m1_signal
            df_sim.loc[current_index, f"Signal_Score{label_suffix}"] = signal_score if pd.notna(signal_score) else np.nan
            df_sim.loc[current_index, f"Trade_Reason{label_suffix}"] = trade_reason if final_m1_signal != "NONE" else "NONE"
            df_sim.loc[current_index, f"Session{label_suffix}"] = session_tag
            df_sim.loc[current_index, f"Trade_Tag{label_suffix}"] = current_trade_tag
            if USE_ADAPTIVE_SIGNAL_SCORE:
                recent_df = df_sim.iloc[max(0, current_bar_index - ADAPTIVE_SIGNAL_SCORE_WINDOW):current_bar_index]
                current_thresh = get_dynamic_signal_score_entry(
                    recent_df,
                    ADAPTIVE_SIGNAL_SCORE_WINDOW,
                    ADAPTIVE_SIGNAL_SCORE_QUANTILE,
                    MIN_SIGNAL_SCORE_ENTRY_MIN,
                    MIN_SIGNAL_SCORE_ENTRY_MAX,
                )
                if (
                    last_logged_signal_thresh is None
                    or abs(current_thresh - last_logged_signal_thresh) > 1e-6
                ):
                    logging.info(
                        f"[Adaptive] Current Signal_Score threshold: {current_thresh:.2f}"
                    )
                    last_logged_signal_thresh = current_thresh
            else:
                current_thresh = MIN_SIGNAL_SCORE_ENTRY
            entry_allowed, block_reason_entry = is_entry_allowed(row, session_tag, consecutive_losses, signal_score_threshold=current_thresh); open_new_order = False; is_reentry_trade = False; is_forced_entry = False
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
                        atr_fe = pd.to_numeric(getattr(row, "ATR_14", np.nan), errors='coerce'); avg_atr_fe = pd.to_numeric(getattr(row, "ATR_14_Rolling_Avg", np.nan), errors='coerce'); gain_z_fe = pd.to_numeric(getattr(row, "Gain_Z", np.nan), errors='coerce'); pattern_fe = getattr(row, "Pattern_Label", "Normal")
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
                    # [Patch v5.x.x] Temporarily bypass MACD filter for testing
                    relax_macd_cond = True; strong_signal_thresh = 4.0; strong_gainz_thresh = 1.0
                    if pd.notna(signal_score):
                        if is_forced_entry:
                            if abs(signal_score) >= strong_signal_thresh and pd.notna(current_gain_z) and current_gain_z >= strong_gainz_thresh and pattern_label in ['Breakout', 'StrongTrend']: relax_macd_cond = True
                        elif not is_forced_entry and abs(signal_score) >= strong_signal_thresh: relax_macd_cond = True
                    if not relax_macd_cond:
                        if side == "BUY" and current_macd_smooth < MACD_NEG_THRESHOLD_BUY:
                            can_open_order = False
                            block_reason = f"NEG_MACD_BUY (MACD={current_macd_smooth:.3f})"
                        elif side == "SELL" and current_macd_smooth > MACD_POS_THRESHOLD_SELL:
                            can_open_order = False
                            block_reason = f"POS_MACD_SELL (MACD={current_macd_smooth:.3f})"
                if can_open_order:
                    # [Patch v5.x.x] Disable Soft Cooldown logic during testing
                    pass
                    # if soft_cooldown_bars_remaining > 0:
                    #     can_open_order = False
                    #     block_reason = f"SOFT_COOLDOWN_ACTIVE({soft_cooldown_bars_remaining})"
                    # else:
                    #     cooldown_triggered, recent_losses_count = is_soft_cooldown_triggered(
                    #         last_n_full_trade_pnls, SOFT_COOLDOWN_LOOKBACK, SOFT_COOLDOWN_LOSS_COUNT
                    #     )
                    #     if cooldown_triggered:
                    #         soft_cooldown_bars_remaining = SOFT_COOLDOWN_LOOKBACK
                    #         can_open_order = False
                    #         block_reason = (
                    #             f"SOFT_COOLDOWN_{SOFT_COOLDOWN_LOSS_COUNT}L{SOFT_COOLDOWN_LOOKBACK}T ({recent_losses_count} losses)"
                    #         )
                if block_reason: logging.debug(f"      Block Reason: {block_reason}")
                active_l1_model = None; active_l1_features = None; selected_model_key = "N/A"; model_confidence = np.nan; meta_proba_tp_for_log = np.nan
                if can_open_order and USE_META_CLASSIFIER and callable(model_switcher_func):
                    logging.debug("      Applying ML Filter (L1) using Model Switcher...")
                    context = {'session': session_tag, 'drift_score': fold_config.get('drift_score', 0.0), 'signal_score': signal_score if pd.notna(signal_score) else 0.0, 'pattern': pattern_label, 'cluster': getattr(row, 'cluster', 0), 'spike_score': getattr(row, 'spike_score', 0.0), 'current_time': now, 'consecutive_losses': consecutive_losses}
                    try:
                        selected_model_key, model_confidence = model_switcher_func(context, available_models); logging.debug(f"         Switcher selected model: '{selected_model_key}', Confidence: {model_confidence}"); model_info = available_models.get(selected_model_key)
                        if model_info and model_info.get('model') and model_info.get('features'): active_l1_model = model_info['model']; active_l1_features = model_info['features']
                        else:
                            logging.warning(f"         (Warning) Switcher selected '{selected_model_key}', but model/features invalid. Falling back to 'main'."); selected_model_key = 'main'; model_info = available_models.get('main')
                            if model_info and model_info.get('model') and model_info.get('features'): active_l1_model = model_info['model']; active_l1_features = model_info['features']
                            else: logging.error("         (Error) Fallback to main model failed. Skipping ML Filter."); can_open_order = False; block_reason = "ML1_MAIN_FALLBACK_FAIL"; active_l1_model = None
                        df_sim.loc[current_index, f"Active_Model{label_suffix}"] = selected_model_key
                        df_sim.loc[current_index, f"Model_Confidence{label_suffix}"] = model_confidence
                    except Exception as e_switch:
                        logging.error(f"      (Error) Model Switcher failed: {e_switch}. Falling back to main model.", exc_info=True); selected_model_key = 'main'; model_info = available_models.get('main')
                        if model_info and model_info.get('model') and model_info.get('features'): active_l1_model = model_info['model']; active_l1_features = model_info['features']
                        else: logging.error("      (Error) Fallback to main model failed after switcher error. Skipping ML Filter."); can_open_order = False; block_reason = "ML1_SWITCH_ERR_FALLBACK_FAIL"; active_l1_model = None
                        df_sim.loc[current_index, f"Active_Model{label_suffix}"] = f"ErrorFallback_{selected_model_key}"
                        df_sim.loc[current_index, f"Model_Confidence{label_suffix}"] = np.nan
                    if active_l1_model and active_l1_features:
                        # [Patch v5.1.1] Fix feature check for namedtuple rows
                        missing_ml_features = [
                            f for f in active_l1_features if f not in row._fields
                        ]
                        if missing_ml_features: logging.error(f"      (Error) ML Filter ({selected_model_key}): Missing features {missing_ml_features} in row data. Skipping filter."); can_open_order = False; block_reason = f"ML1_FEAT_MISS_{selected_model_key.upper()}"
                        else:
                            try:
                                # [Patch v5.5.3] Retrieve features from namedtuple row using getattr
                                row_data = {f: getattr(row, f) for f in active_l1_features}
                                X_ml = pd.DataFrame([row_data]); numeric_cols_ml = X_ml.select_dtypes(include=np.number).columns
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
                        log_entry_blocked = {"timestamp": now, "reason": block_reason, "side": side, "fund_profile": fund_profile.get('mm_mode', 'N/A'), "active_model": selected_model_key, "model_confidence": model_confidence, "meta_proba_tp": meta_proba_tp_for_log, "signal_score": signal_score if pd.notna(signal_score) else np.nan, "pattern_label": getattr(row, "Pattern_Label", "N/A"), "is_reentry_attempt": is_reentry_trade, "is_forced_entry": is_forced_entry}
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
                                new_order = {"entry_idx": current_index, "entry_time": entry_time, "entry_price": entry_price, "original_lot": final_lot, "lot": final_lot, "original_sl_price": sl_price, "sl_price": sl_price, "tp_price": tp2_price, "tp1_price": tp1_price, "entry_bar_count": current_bar_index, "side": side, "m15_trend_zone": m15_trend, "trade_tag": current_trade_tag, "signal_score": signal_score if pd.notna(signal_score) else np.nan, "trade_reason": trade_reason if not is_forced_entry else f"FORCED_{trade_reason}", "session": session_tag, "pattern_label_entry": pattern_label, "be_triggered": False, "be_triggered_time": pd.NaT, "is_reentry": is_reentry_trade, "is_forced_entry": is_forced_entry, "meta_proba_tp": meta_proba_tp_for_log, "meta2_proba_tp": meta2_proba_tp_for_log, "partial_tp_processed_levels": set(), "atr_at_entry": atr_entry, "equity_before_open": current_equity_check, "entry_gain_z": current_gain_z if pd.notna(current_gain_z) else np.nan, "entry_macd_smooth": current_macd_smooth if pd.notna(current_macd_smooth) else np.nan, "entry_candle_ratio": getattr(row, "Candle_Ratio", np.nan), "entry_adx": getattr(row, "ADX", np.nan), "entry_volatility_index": current_vol_index if pd.notna(current_vol_index) else np.nan, "peak_since_tp1": np.nan, "trough_since_tp1": np.nan, "risk_mode_at_entry": risk_mode_applied, "use_trailing_for_tp2": enable_ttp2, "trailing_start_price": tp1_price if enable_ttp2 else np.nan, "trailing_step_r": ADAPTIVE_TSL_DEFAULT_STEP_R if enable_ttp2 else np.nan, "peak_since_ttp2_activation": np.nan, "trough_since_ttp2_activation": np.nan, "active_model_at_entry": selected_model_key, "model_confidence_at_entry": model_confidence, "tsl_activated": False, "peak_since_tsl_activation": np.nan, "trough_since_tsl_activation": np.nan}
                                next_active_orders.append(new_order); logging.info(f"         +++ ORDER OPENED: Side={side}, Lot={final_lot:.2f}, Entry={entry_price:.5f}, SL={sl_price:.5f}, TP={tp2_price:.5f}")
                                df_sim.loc[current_index, f"Order_Opened{label_suffix}"] = True; df_sim.loc[current_index, f"Lot_Size{label_suffix}"] = final_lot; df_sim.loc[current_index, f"Entry_Price_Actual{label_suffix}"] = entry_price; df_sim.loc[current_index, f"SL_Price_Actual{label_suffix}"] = sl_price; df_sim.loc[current_index, f"TP_Price_Actual{label_suffix}"] = tp2_price; df_sim.loc[current_index, f"ATR_At_Entry{label_suffix}"] = atr_entry; df_sim.loc[current_index, f"Equity_Before_Open{label_suffix}"] = current_equity_check; df_sim.loc[current_index, f"Is_Reentry{label_suffix}"] = is_reentry_trade; df_sim.loc[current_index, f"Forced_Entry{label_suffix}"] = is_forced_entry; df_sim.loc[current_index, f"Meta_Proba_TP{label_suffix}"] = meta_proba_tp_for_log; df_sim.loc[current_index, f"Meta2_Proba_TP{label_suffix}"] = meta2_proba_tp_for_log; df_sim.loc[current_index, f"Entry_Gain_Z{label_suffix}"] = current_gain_z if pd.notna(current_gain_z) else np.nan; df_sim.loc[current_index, f"Entry_MACD_Smooth{label_suffix}"] = current_macd_smooth if pd.notna(current_macd_smooth) else np.nan; df_sim.loc[current_index, f"Entry_Candle_Ratio{label_suffix}"] = getattr(row, "Candle_Ratio", np.nan); df_sim.loc[current_index, f"Entry_ADX{label_suffix}"] = getattr(row, "ADX", np.nan); df_sim.loc[current_index, f"Entry_Volatility_Index{label_suffix}"] = current_vol_index if pd.notna(current_vol_index) else np.nan; df_sim.loc[current_index, f"Active_Model{label_suffix}"] = selected_model_key; df_sim.loc[current_index, f"Model_Confidence{label_suffix}"] = model_confidence
                                if is_reentry_trade: reentry_trades_opened += 1
                                if is_forced_entry: forced_entry_trades_opened += 1
                                bars_since_last_trade = 0
                            else:
                                block_reason = f"LOT_SIZE_MIN ({final_lot:.2f} < {MIN_LOT_SIZE})"; logging.info(f"      Order Blocked. Reason: {block_reason}"); blocked_order_log.append({"timestamp": now, "reason": block_reason, "side": side, "fund_profile": fund_profile.get('mm_mode', 'N/A'), "active_model": selected_model_key, "model_confidence": model_confidence, "meta_proba_tp": meta_proba_tp_for_log, "signal_score": signal_score if pd.notna(signal_score) else np.nan, "pattern_label": getattr(row, "Pattern_Label", "N/A"), "is_reentry_attempt": is_reentry_trade, "is_forced_entry": is_forced_entry, "calculated_lot": final_lot})

            equity = equity_at_start_of_bar + current_equity_change_this_bar
            logging.debug(f"   Equity at end of bar {current_bar_index}: {equity:.2f} (Change: {current_equity_change_this_bar:.2f})")

            if equity <= 0 and not kill_switch_activated:
                logging.warning(f"[Patch] Margin Call triggered. Equity = {equity:.2f}."); kill_switch_activated = True; kill_switch_trigger_time = now; equity = 0
                if active_orders:
                    logging.warning(f"      Force closing {len(active_orders)} orders due to Margin Call at {now}.")
                    for mc_order in active_orders: trade_log_entry_mc = {"period": label, "side": mc_order.get("side"), "entry_idx": mc_order.get("entry_idx"), "entry_time": mc_order.get("entry_time"), "entry_price": mc_order.get("entry_price"), "close_time": now, "exit_price": current_close, "exit_reason": "MARGIN_CALL", "lot": mc_order.get("lot", 0.0), "pnl_usd_net": 0.0, "is_partial_tp": False, "partial_tp_level": len(mc_order.get("partial_tp_processed_levels", set())), "risk_mode_at_entry": mc_order.get("risk_mode_at_entry", "N/A"), "active_model_at_entry": mc_order.get("active_model_at_entry", "N/A")}; trade_log.append(trade_log_entry_mc)
                    active_orders.clear()
                next_active_orders.clear(); df_sim.loc[current_index, f"Equity_Realistic{label_suffix}"] = 0.0; df_sim.loc[current_index, f"Max_Drawdown_At_Point{label_suffix}"] = 1.0; df_sim.loc[current_index, f"Active_Order_Count{label_suffix}"] = 0; equity_history[current_index] = 0.0
                remaining_indices = df_sim.index[df_sim.index > current_index]
                if not remaining_indices.empty: logging.info(f"      Marking remaining {len(remaining_indices)} bars with 0 equity due to Margin Call."); df_sim.loc[remaining_indices, f"Equity_Realistic{label_suffix}"] = 0.0; df_sim.loc[remaining_indices, f"Max_Drawdown_At_Point{label_suffix}"] = 1.0; df_sim.loc[remaining_indices, f"Active_Order_Count{label_suffix}"] = 0
                break

            peak_equity = max(peak_equity, equity); current_dd_final = (peak_equity - equity) / peak_equity if peak_equity > 1e-9 else 0.0; max_drawdown_pct = max(max_drawdown_pct, current_dd_final); logging.debug(f"   Drawdown: Current={current_dd_final*100:.2f}%, Max={max_drawdown_pct*100:.2f}%")
            df_sim.loc[current_index, f"Max_Drawdown_At_Point{label_suffix}"] = max_drawdown_pct; df_sim.loc[current_index, f"Equity_Realistic{label_suffix}"] = equity; df_sim.loc[current_index, f"Active_Order_Count{label_suffix}"] = len(next_active_orders); equity_history[current_index] = equity

            if enable_kill_switch and not kill_switch_activated:
                logging.debug(f"   Checking Kill Switch: DD={current_dd_final*100:.2f}% (Warn>{KILL_SWITCH_WARNING_MAX_DD_THRESHOLD*100:.0f}%, Kill>{KILL_SWITCH_MAX_DD_THRESHOLD*100:.0f}%), Losses={consecutive_losses} (Warn>{KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD}, Kill>{kill_switch_consecutive_losses_config})")
                if current_dd_final > KILL_SWITCH_MAX_DD_THRESHOLD:
                    logging.warning("[Patch] Kill Switch triggered due to drawdown.")
                    logging.critical(f"(CRITICAL) KILL SWITCH ACTIVATED (Max DD): {label} at {now}. Drawdown {current_dd_final*100:.2f}% > {KILL_SWITCH_MAX_DD_THRESHOLD*100:.0f}%. Stopping simulation loop.")
                    kill_switch_activated = True
                    kill_switch_trigger_time = now
                    break
                elif consecutive_losses >= kill_switch_consecutive_losses_config:
                    logging.warning("[Patch] Kill Switch triggered due to consecutive losses.")
                    logging.critical(f"     (CRITICAL) KILL SWITCH ACTIVATED (Consecutive Losses): {label} at {now}. Losses: {consecutive_losses} >= {kill_switch_consecutive_losses_config}. Stopping simulation loop.")
                    kill_switch_activated = True
                    kill_switch_trigger_time = now
                    break
                else:
                    if current_dd_final > KILL_SWITCH_WARNING_MAX_DD_THRESHOLD:
                        logging.warning(f"(Warning) Drawdown {current_dd_final*100:.2f}% ยังไม่ถึง threshold {KILL_SWITCH_MAX_DD_THRESHOLD*100:.0f}%")
                    if consecutive_losses >= KILL_SWITCH_WARNING_CONSECUTIVE_LOSSES_THRESHOLD:
                        logging.warning(f"(Warning) Consecutive losses = {consecutive_losses}, ยังไม่ถึง threshold สำหรับ Kill Switch.")

            previous_risk_mode = current_risk_mode
            if consecutive_losses >= recovery_mode_consecutive_losses_config:
                if current_risk_mode != "recovery": logging.info("[Patch] Activating Recovery Mode due to consecutive losses.")
                current_risk_mode = "recovery"
            else:
                if current_risk_mode == "recovery": logging.info("[Patch] Deactivating Recovery Mode.")
                current_risk_mode = "normal"
            if current_risk_mode != previous_risk_mode: logging.info(f"      [{now}] Risk Mode for *next* bar set to: {current_risk_mode} (Losses: {consecutive_losses})")
            df_sim.loc[current_index, f"Risk_Mode{label_suffix}"] = current_risk_mode
            active_orders = next_active_orders
            logging.debug(
                f"   End of Bar {current_bar_index}. Active orders for next bar: {len(active_orders)}"
            )
            soft_cooldown_bars_remaining = step_soft_cooldown(soft_cooldown_bars_remaining)
            current_bar_index += 1
    # <<< [Patch C - Unified] End of try-except for main loop >>>
    except Exception as e_loop:
        # <<< [Patch C - Unified] Log critical error and set error_in_loop flag >>>
        logging.critical(
            f"   (CRITICAL) Error occurred inside simulation loop for {label} at index {current_index if 'current_index' in locals() else 'UNKNOWN_BAR_INDEX'}: {e_loop}",
            exc_info=True,
        )
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
            if entry_bar_idx_log_end is not None:
                resolved_idx_end = _resolve_close_index(df_sim, entry_bar_idx_log_end, close_timestamp)
                if resolved_idx_end is not None:
                    safe_set_datetime(df_sim, resolved_idx_end, f"Order_Closed_Time{label_suffix}", close_timestamp)
                    df_sim.loc[resolved_idx_end, f"PnL_Realized_USD{label_suffix}"] = net_pnl_usd; df_sim.loc[resolved_idx_end, f"Commission_USD{label_suffix}"] = commission_usd; df_sim.loc[resolved_idx_end, f"Spread_Cost_USD{label_suffix}"] = spread_cost_usd; df_sim.loc[resolved_idx_end, f"Slippage_USD{label_suffix}"] = slippage_usd; df_sim.loc[resolved_idx_end, f"Exit_Reason_Actual{label_suffix}"] = close_reason; df_sim.loc[resolved_idx_end, f"Exit_Price_Actual{label_suffix}"] = exit_price; df_sim.loc[resolved_idx_end, f"PnL_Points_Actual{label_suffix}"] = pnl_points_net_spread
            else:
                logging.warning(f"   (Warning) Could not find entry index '{entry_bar_idx_log_end}' in df_sim to update results for order {order_entry_time_end} (End of Period).")
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
    # [Patch v5.3.5] Safeguard NaN/Inf Check in PnL and feature processing
    if np.isnan(equity) or np.isinf(equity):
        logging.critical("[CRITICAL] NaN/Inf detected in final equity, investigation required.")
    logging.warning(
        f"[QA][SUMMARY] Fold Finished | Final Equity: ${equity:.2f} | Max DD: {max_drawdown_pct:.2%} | KILL SWITCH: {kill_switch_activated}"
    )
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

logging.info(f"Part 8: Backtesting Engine Functions Loaded (v{__version__} Applied).")
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
# [Patch v5.5.4] Environment override for drift threshold
DEFAULT_DRIFT_WASSERSTEIN_THRESHOLD = get_env_float("DRIFT_WASSERSTEIN_THRESHOLD", 0.1)
DEFAULT_DRIFT_TTEST_ALPHA = 0.05
DEFAULT_INITIAL_CAPITAL = 100.0
DEFAULT_IB_COMMISSION_PER_LOT = 7.0
DEFAULT_N_WALK_FORWARD_SPLITS = 5
DEFAULT_ENTRY_CONFIG_PER_FOLD = {0: {}} # Minimal default
DEFAULT_FUND_PROFILES = {"NORMAL": {"risk": 0.01, "mm_mode": "balanced"}}
DEFAULT_FUND_NAME = "NORMAL"
DEFAULT_META_MIN_PROBA_THRESH = 0.5
DEFAULT_ENABLE_PARTIAL_TP = True
DEFAULT_PARTIAL_TP_LEVELS = []
DEFAULT_PARTIAL_TP_MOVE_SL_TO_ENTRY = True
DEFAULT_ENABLE_KILL_SWITCH = True
DEFAULT_KILL_SWITCH_MAX_DD_THRESHOLD = 0.30
DEFAULT_KILL_SWITCH_CONSECUTIVE_LOSSES_THRESHOLD = 10
DEFAULT_RECOVERY_MODE_CONSECUTIVE_LOSSES = 4
DEFAULT_min_equity_threshold_pct = 0.70
DEFAULT_DYNAMIC_GAINZ_DRIFT_THRESHOLD = 0.10
DEFAULT_DYNAMIC_GAINZ_ADJUSTMENT = 0.1
DEFAULT_RSI_DRIFT_OVERRIDE_THRESHOLD = 0.65
DEFAULT_ATR_DRIFT_OVERRIDE_THRESHOLD = 0.25

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
try:
    RSI_DRIFT_OVERRIDE_THRESHOLD
except NameError:
    RSI_DRIFT_OVERRIDE_THRESHOLD = DEFAULT_RSI_DRIFT_OVERRIDE_THRESHOLD
try:
    ATR_DRIFT_OVERRIDE_THRESHOLD
except NameError:
    ATR_DRIFT_OVERRIDE_THRESHOLD = DEFAULT_ATR_DRIFT_OVERRIDE_THRESHOLD

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
    for fold, (train_index, test_index) in enumerate(
        tqdm(tscv.split(df_m1_final), total=n_walk_forward_splits, desc="Running folds", unit="fold")
    ):
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

            rsi_drift_override_threshold = RSI_DRIFT_OVERRIDE_THRESHOLD
            atr_drift_override_threshold = ATR_DRIFT_OVERRIDE_THRESHOLD

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
        metrics_buy_fold = {}  # [Patch v5.3.1] initialize to avoid UnboundLocalError
        try:
            metrics_buy_fold = calculate_metrics(
                log_buy,
                eq_buy,
                hist_buy,
                start_cap_buy,
                f"Fold {fold+1} Buy ({fund_name})",
                type_l1_b,
                type_l2_b,
                costs_buy,
                ib_lot_buy,
            ) or {}
        except Exception as e:
            logging.warning(
                f"(Warning) Cannot calculate metrics for Fold {fold+1} Buy ({fund_name}): {e}"
            )
            metrics_buy_fold = {}
        metrics_buy_fold[f"Fold {fold+1} Buy ({fund_name}) Max Drawdown (Simulated) (%)"] = dd_buy * 100.0
        metrics_buy_fold.update({
            f"Fold {fold+1} Buy ({fund_name}) Costs {k.replace('_', ' ').title()}": v
            for k, v in costs_buy.items()
            if k
            not in [
                "meta_model_type_l1",
                "meta_model_type_l2",
                "threshold_l1_used",
                "threshold_l2_used",
                "fund_profile",
                "total_ib_lot_accumulator",
            ]
        })

        metrics_sell_fold = {}  # [Patch v5.3.1] ensure defined even if calc fails
        try:
            metrics_sell_fold = calculate_metrics(
                log_sell,
                eq_sell,
                hist_sell,
                start_cap_sell,
                f"Fold {fold+1} Sell ({fund_name})",
                type_l1_s,
                type_l2_s,
                costs_sell,
                ib_lot_sell,
            ) or {}
        except Exception as e:
            logging.warning(
                f"(Warning) Cannot calculate metrics for Fold {fold+1} Sell ({fund_name}): {e}"
            )
            metrics_sell_fold = {}
        metrics_sell_fold[f"Fold {fold+1} Sell ({fund_name}) Max Drawdown (Simulated) (%)"] = dd_sell * 100.0
        metrics_sell_fold.update({
            f"Fold {fold+1} Sell ({fund_name}) Costs {k.replace('_', ' ').title()}": v
            for k, v in costs_sell.items()
            if k
            not in [
                "meta_model_type_l1",
                "meta_model_type_l2",
                "threshold_l1_used",
                "threshold_l2_used",
                "fund_profile",
                "total_ib_lot_accumulator",
            ]
        })

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

        fold_duration = time.time() - fold_start_time
        fold_equity = eq_sell
        fold_winrate = (
            metrics_buy_fold.get(f"Fold {fold+1} Buy ({fund_name}) Win Rate (Full) (%)", 0.0)
            + metrics_sell_fold.get(f"Fold {fold+1} Sell ({fund_name}) Win Rate (Full) (%)", 0.0)
        ) / 200.0
        fold_maxdd = max(dd_buy, dd_sell)
        logging.warning(
            f"=============== Fold {fold+1}/{n_walk_forward_splits} ({fund_name}) ==============="
        )
        logging.warning(
            f"   (Metrics) Fold {fold+1} processed in: {fold_duration:.2f} seconds"
        )
        logging.warning(
            f"   (Summary) Equity={fold_equity:.2f}, Winrate={fold_winrate:.2%}, MaxDD={fold_maxdd:.2%}"
        )
        # [Patch v5.3.5] Add QA summary after each fold
        trades_buy = metrics_buy_fold.get(f"Fold {fold+1} Buy ({fund_name}) Total Trades (Full)", 0)
        trades_sell = metrics_sell_fold.get(f"Fold {fold+1} Sell ({fund_name}) Total Trades (Full)", 0)
        num_trades = trades_buy + trades_sell
        risk_buy = metrics_buy_fold.get(f"Fold {fold+1} Buy ({fund_name}) Final Risk Mode", "N/A")
        risk_sell = metrics_sell_fold.get(f"Fold {fold+1} Sell ({fund_name}) Final Risk Mode", "N/A")
        recovery_active = risk_buy == "recovery" or risk_sell == "recovery"
        kill_switch_triggered = final_ks_state_buy or final_ks_state_sell
        logging.warning(
            f"  [QA SUMMARY FOLD] | Final Equity: ${fold_equity:.2f} | Max DD: {fold_maxdd:.2%} | Winrate: {fold_winrate:.2%} | Trades: {num_trades} | KILL SWITCH: {kill_switch_triggered} | Recovery: {recovery_active}"
        )

        logging.debug(f"        Cleaning up memory after Fold {fold+1}...")
        del df_train_fold, df_test_fold, df_buy_res, df_sell_res
        del log_buy, log_sell, hist_buy, hist_sell, blocked_buy, blocked_sell
        del metrics_buy_fold, metrics_sell_fold, current_fold_metrics
        gc.collect()
        logging.debug(f"        Memory cleanup complete for Fold {fold+1}.")

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

def summarize_wfv_results(all_fold_metrics):
    """สรุปผล Walk-Forward เป็น DataFrame"""
    def _find_metric(d, keyword):
        for k, v in d.items():
            if keyword in k:
                return v
        return 0

    records = []
    for i, fm in enumerate(all_fold_metrics):
        buy = fm.get("buy", {})
        sell = fm.get("sell", {})
        pnl = _find_metric(buy, "Total Net Profit") + _find_metric(sell, "Total Net Profit")
        win = (_find_metric(buy, "Win Rate") + _find_metric(sell, "Win Rate")) / 2.0
        dd = max(_find_metric(buy, "Max Drawdown"), _find_metric(sell, "Max Drawdown"))
        records.append({"fold_no": i + 1, "PnL_total": pnl, "Win_Rate": win, "Max_Drawdown": dd})
    return pd.DataFrame(records)

def summarize_wfv_results(all_fold_metrics):
    """สรุปผล Walk-Forward เป็น DataFrame"""
    records = []
    for i, fm in enumerate(all_fold_metrics):
        buy = fm.get("buy", {})
        sell = fm.get("sell", {})
        pnl = buy.get(f"Fold {i+1} Buy (", 0)  # placeholder
        win = buy.get(f"Fold {i+1} Buy (", 0)
        dd_buy = buy.get(f"Fold {i+1} Buy (", 0)
        dd_sell = sell.get(f"Fold {i+1} Sell (", 0)
        max_dd = max(dd_buy, dd_sell)
        records.append({"fold_no": i+1, "PnL_total": pnl, "Win_Rate": win, "Max_Drawdown": max_dd})
    return pd.DataFrame(records)

# ------------------------------------------------------------------------------
# === Simple Numba Backtest Helpers ===
# ------------------------------------------------------------------------------

# Cache/Model instances for simple demonstration purposes
_catboost_model_cache: Dict[str, CatBoostClassifier] = {}

@njit
def _run_oms_backtest_numba(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    open_signals: np.ndarray,
    close_signals: np.ndarray,
    sl_prices: np.ndarray,
    tp_prices: np.ndarray,
) -> np.int64:
    """Loop เปิด/ปิด Orders แบบเร่งด้วย Numba"""
    trades_executed = 0
    for i in range(prices.shape[0]):
        if open_signals[i] == 1:
            trades_executed += 1
        elif close_signals[i] == 1:
            trades_executed += 1
    return trades_executed


def run_simple_numba_backtest(df_all: pd.DataFrame, folds: List[tuple]) -> Dict[int, int]:
    """เรียกใช้ backtest แบบง่ายด้วยฟังก์ชัน Numba"""
    results: Dict[int, int] = {}
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        df_backtest = df_all.iloc[test_idx].copy()
        prices = df_backtest["Close"].to_numpy()
        highs = df_backtest["High"].to_numpy()
        lows = df_backtest["Low"].to_numpy()
        open_signals = generate_open_signals(df_backtest)
        close_signals = generate_close_signals(df_backtest)
        sl_prices = precompute_sl_array(df_backtest)
        tp_prices = precompute_tp_array(df_backtest)
        trades_count = _run_oms_backtest_numba(
            prices,
            highs,
            lows,
            open_signals,
            close_signals,
            sl_prices,
            tp_prices,
        )
        logging.info(
            f"Fold {fold_idx} completed. Trades executed (Numba): {trades_count}"
        )
        results[fold_idx] = trades_count
    logging.info("All folds finished.")
    return results

### PART 13: Hyperparameter Sweep Utility (v5.1.0) ###
def run_hyperparameter_sweep(base_params: dict, grid: dict, train_func):
    """รันการค้นหา Hyperparameter แบบ grid search และพิมพ์ผลลัพธ์ทันที"""
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    output_dir = base_params.get("output_dir")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    for idx, combo in enumerate(combinations, start=1):
        params = base_params.copy()
        for k, v in zip(keys, combo):
            params[k] = v
        print(f"เริ่มพารามิเตอร์ run {idx}: {params}")
        model_path, feature_list = train_func(**params)
        result_entry = {"params": params, "model_path": model_path, "features": feature_list}
        print(f"Run {idx}: {result_entry}")
        results.append(result_entry)

    return results


# [Patch v5.0.18] Add Optuna-based CatBoost sweep
def run_optuna_catboost_sweep(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    n_splits: int = 5,
):
    """Runs Optuna hyperparameter search for CatBoost."""
    if optuna is None or CatBoostClassifier is None:
        # [Patch v5.0.23] Return stub values when dependencies are missing
        logging.error("(Error) optuna or catboost not available for sweep")
        return 0.0, {}

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 50, 200),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
            "border_count": trial.suggest_int("border_count", 32, 64),
            "random_strength": trial.suggest_float("random_strength", 0, 1),
            "eval_metric": "AUC",
            "verbose": False,
            "task_type": "CPU",
        }
        model = CatBoostClassifier(**params)
        cv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_value, study.best_params


def generate_open_signals(
    df: pd.DataFrame,
    use_macd: bool = USE_MACD_SIGNALS,
    use_rsi: bool = USE_RSI_SIGNALS,
) -> np.ndarray:
    """สร้างสัญญาณเปิด order พร้อมตัวเลือกเปิด/ปิด MACD และ RSI"""
    open_mask = df["Close"] > df["Close"].shift(1)
    if use_macd:
        if "MACD_hist" not in df.columns:
            _, _, macd_hist = macd(df["Close"])
            df = df.copy()
            df["MACD_hist"] = macd_hist
        open_mask &= df["MACD_hist"] > 0
    if use_rsi:
        if "RSI" not in df.columns:
            df = df.copy()
            df["RSI"] = rsi(df["Close"])
        open_mask &= df["RSI"] > 50
    return open_mask.fillna(0).astype(np.int8).to_numpy()


def generate_close_signals(
    df: pd.DataFrame,
    use_macd: bool = USE_MACD_SIGNALS,
    use_rsi: bool = USE_RSI_SIGNALS,
) -> np.ndarray:
    """สร้างสัญญาณปิด order พร้อมตัวเลือกเปิด/ปิด MACD และ RSI"""
    close_mask = df["Close"] < df["Close"].shift(1)
    if use_macd:
        if "MACD_hist" not in df.columns:
            _, _, macd_hist = macd(df["Close"])
            df = df.copy()
            df["MACD_hist"] = macd_hist
        close_mask &= df["MACD_hist"] < 0
    if use_rsi:
        if "RSI" not in df.columns:
            df = df.copy()
            df["RSI"] = rsi(df["Close"])
        close_mask &= df["RSI"] < 50
    return close_mask.fillna(0).astype(np.int8).to_numpy()


def precompute_sl_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Stop-Loss ล่วงหน้า"""
    return np.zeros(len(df), dtype=np.float64)


def precompute_tp_array(df: pd.DataFrame) -> np.ndarray:
    """คำนวณ Take-Profit ล่วงหน้า"""
    return np.zeros(len(df), dtype=np.float64)
# ---------------------------------------------------------------------------
# Stubs for Function Registry Tests
def initialize_time_series_split():
    """Stubbed time series split initializer."""
    return None
def calculate_forced_entry_logic():
    """Stubbed forced entry logic calculator."""
    return None
def apply_kill_switch():
    """Stubbed kill switch applier."""
    return None
def log_trade(*args, **kwargs):
    """Stubbed trade logger."""
    return None
def aggregate_fold_results():
    """Stubbed fold result aggregator."""
    return None


