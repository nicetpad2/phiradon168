"""Bootstrap script for running the main entry point."""

# [Patch v6.4.0] Ensure project modules are importable by setting sys.path and working directory
import sys
import os
from pathlib import Path

# Add project root to PYTHONPATH and set cwd
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

import logging
from src.features import DEFAULT_META_CLASSIFIER_FEATURES

# [Patch v6.3.0] Stub imports for missing features
try:
    from src.utils.auto_train_meta_classifiers import auto_train_meta_classifiers
except ImportError:  # pragma: no cover - fallback when module missing

    def auto_train_meta_classifiers(*args, **kwargs):
        logging.getLogger().warning(
            "[Patch v6.2.3] auto_train_meta_classifiers stub invoked; skipping."
        )


try:
    from reporting.dashboard import generate_dashboard
except ImportError:  # pragma: no cover - fallback when module missing

    def generate_dashboard(*args, **kwargs):
        logging.getLogger().warning(
            "[Patch v6.2.3] generate_dashboard stub invoked; skipping."
        )


import csv
import yaml
import time
from pathlib import Path

try:  # [Patch v5.10.2] allow import without heavy dependencies
    from src.config import logger, OUTPUT_DIR, DEFAULT_TRADE_LOG_MIN_ROWS
    import src.config as config
except Exception:  # pragma: no cover - fallback logger for tests
    logger = logging.getLogger("ProjectP")
    OUTPUT_DIR = Path("output_default")
# [Patch v5.9.17] Fallback logger if src.config fails
import sys
import os
import argparse
import subprocess
import json

from src.utils.pipeline_config import (
    load_config,
)  # [Patch v6.7.17] dynamic config loader

# [Patch v6.4.8] Optional fallback directory for raw data and logs
FALLBACK_DIR = os.getenv("PROJECTP_FALLBACK_DIR")

# [Patch v6.7.17] Load pipeline configuration for dynamic paths
pipeline_config = load_config()


# [Patch v6.3.1] Ensure working directory fallback on import
try:
    os.getcwd()
except Exception:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    print(f"[Info] Changed working directory to project root: {project_root}")

import pandas as pd
from typing import Dict, List
from src.utils.errors import PipelineError
import main as pipeline
from config_loader import update_config_from_dict  # [Patch] dynamic config update
from wfv_runner import run_walkforward  # [Patch] walk-forward helper
from src.features import build_feature_catalog
try:
    from main import run_preprocess as pipeline_run_preprocess
except Exception:  # pragma: no cover - fallback when test stubs main
    def pipeline_run_preprocess(*_a, **_kw):
        logging.getLogger().warning(
            "[Patch] pipeline_run_preprocess stub invoked"
        )

# Default grid for hyperparameter sweep
DEFAULT_SWEEP_PARAMS: Dict[str, List[float]] = {
    "learning_rate": [0.01, 0.05],
    "depth": [6, 8],
    "l2_leaf_reg": [1, 3, 5],
    "subsample": [0.8, 1.0],
    "colsample_bylevel": [0.8, 1.0],
    "bagging_temperature": [0.0, 1.0],
    "random_strength": [0.0, 1.0],
}

# [Patch] Initialize pynvml for GPU status detection
try:
    import pynvml

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None
    nvml_handle = None
except Exception:  # pragma: no cover - NVML failure fallback
    nvml_handle = None

from src.main import main, setup_output_directory, OUTPUT_BASE_DIR, OUTPUT_DIR_NAME
from src.data_loader import (
    auto_convert_gold_csv as auto_convert_csv,
    auto_convert_csv_to_parquet,
)
from src.utils.model_utils import get_latest_model_and_threshold


def configure_logging():
    """Set up consistent logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_logo() -> None:
    """แสดงโลโก้ Project P บนหน้าจอ."""
    logo = r"""
 ____            _            ____  ____  
|  _ \ ___  __ _| | _____    |  _ \|  _ \ 
| |_) / _ \/ _` | |/ / _ \   | |_) | |_) |
|  __/  __/ (_| |   <  __/   |  __/|  __/ 
|_|   \___|\__,_|_|\_\___|   |_|   |_|    
    """
    print(logo)


def custom_helper_function():
    """Stubbed helper for tests."""
    return True


def interactive_menu() -> str:
    """แสดงเมนูโต้ตอบและคืนค่าโหมดที่เลือก."""
    options = {
        "1": "full_pipeline",
        "2": "preprocess",
        "3": "sweep",
        "4": "threshold",
        "5": "backtest",
        "6": "report",
        "7": "wfv",
        "0": "exit",
    }
    for key, val in options.items():
        print(f"[{key}] {val}")
    choice = input("เลือกโหมด: ").strip()
    return options.get(choice)


def parse_projectp_args(args=None):
    """Parse command line arguments for ProjectP."""
    parser = argparse.ArgumentParser(description="สคริปต์ควบคุมโหมดการทำงาน")
    parser.add_argument(
        "--mode",
        choices=[
            "preprocess",
            "sweep",
            "threshold",
            "backtest",
            "report",
            "full_pipeline",
            "all",
            "hyper_sweep",
            "wfv",
        ],
        default="preprocess",
        help="ขั้นตอนที่จะรัน",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (limit rows for fast pipeline loop test)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Limit number of rows loaded from data (overrides debug default)",
    )
    parser.add_argument(
        "--auto-convert",
        action="store_true",
        help="แปลงไฟล์ CSV อัตโนมัติ",
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="แสดงเมนูโต้ตอบเพื่อเลือกโหมด",
    )
    return parser.parse_known_args(args)[0]


def parse_args(args=None):  # backward compatibility
    return parse_projectp_args(args)


def run_preprocess():
    """รันขั้นตอนเตรียมข้อมูลและฝึกโมเดล."""
    # [Patch v6.9.42] Set flag to avoid recursive subprocess calls
    os.environ["FROM_PROJECTP"] = "1"
    try:
        pipeline_run_preprocess(pipeline_config)
    finally:
        os.environ.pop("FROM_PROJECTP", None)


def _run_script(relative_path: str) -> None:
    """Execute a Python script located relative to this file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, relative_path)
    subprocess.run([sys.executable, abs_path], check=True)


def run_hyperparameter_sweep(
    params: Dict[str, List[float]], trade_log_path: str | None = None
) -> None:
    """รันการค้นหาค่าพารามิเตอร์."""
    logger.debug(f"Starting sweep with params: {params}")
    from tuning.hyperparameter_sweep import run_sweep as _sweep, DEFAULT_TRADE_LOG

    m1_path = os.path.join(str(OUTPUT_DIR), "final_data_m1_v32_walkforward.csv.gz")
    logger.info("[Patch v6.6.6] Running sweep with m1_path=%s", m1_path)

    trade_log = trade_log_path or DEFAULT_TRADE_LOG

    _sweep(
        str(OUTPUT_DIR),
        params,
        seed=42,
        resume=True,
        trade_log_path=trade_log,
        m1_path=m1_path,
    )


def run_sweep():
    """รันการค้นหาค่าพารามิเตอร์ (backward compatibility)."""
    _run_script(os.path.join("tuning", "hyperparameter_sweep.py"))


def run_threshold_optimization() -> pd.DataFrame:
    """รันการปรับค่า threshold."""
    logger.debug("Starting threshold optimization")
    from threshold_optimization import run_threshold_optimization as _opt

    return _opt()


def run_threshold():
    """รันการปรับค่า threshold (backward compatibility)."""
    _run_script("threshold_optimization.py")


def run_backtest():
    """รันการทดสอบย้อนหลัง."""
    model_dir = "models"
    model_path, threshold = get_latest_model_and_threshold(
        model_dir, "threshold_wfv_optuna_results.csv", take_first=True
    )
    pipeline.run_backtest_pipeline(
        pd.DataFrame(), pd.DataFrame(), model_path, threshold
    )


def run_report() -> None:
    """สร้างรายงานผลการทดสอบ."""
    config = pipeline.load_config()
    pipeline.run_report(config)


def _execute_step(name: str, func, *args, **kwargs):
    """[Patch v6.9.23] Execute a pipeline step with timing and logs."""
    start = time.perf_counter()
    logger.info("[Patch v6.9.23] Starting %s", name)
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    logger.warning("[Patch v6.9.23] %s completed in %.2fs", name, elapsed)
    logging.getLogger().warning("%s completed in %.2fs", name, elapsed)
    return result


def run_full_pipeline() -> None:
    """รันทุกโหมดต่อเนื่องกันแบบครบวงจร (เทพเวอร์ชัน)"""
    # Step 1: Preprocess Data
    _execute_step("preprocess", run_preprocess)

    trade_log_default = os.path.join(
        str(OUTPUT_DIR), "trade_log_v32_walkforward.csv.gz"
    )
    try:
        trade_log_path = ensure_trade_log(trade_log_default)
    except FileNotFoundError:
        logger.error("Trade log missing and could not be generated. Aborting.")
        return

    # Step 2: Hyperparameter Sweep
    _execute_step(
        "sweep",
        run_hyperparameter_sweep,
        DEFAULT_SWEEP_PARAMS,
        trade_log_path,
    )

    # Step 3: Auto-apply best hyperparameters from sweep
    logger.info("Applying best hyperparameters found from sweep...")
    best_params_path = os.path.join(str(OUTPUT_DIR), "best_param.json")
    if os.path.exists(best_params_path):
        with open(best_params_path, "r", encoding="utf-8") as fh:
            best_params = json.load(fh)
        update_config_from_dict(best_params)
        logger.warning(f"Successfully applied best hyperparameters from: {best_params_path}")
    else:
        logger.error(
            f"Hyperparameter result file 'best_param.json' not found in {OUTPUT_DIR}. "
            "Cannot apply best params. Continuing with default parameters."
        )

    # Step 4: Threshold Optimization
    _execute_step("threshold", run_threshold_optimization)

    # Step 5: Walk-Forward Validation (Replaced Backtest for a more robust evaluation)
    logger.warning("Replacing standard backtest with robust Walk-Forward Validation (WFV)...")
    _execute_step("walk-forward", run_walkforward)

    # Step 6 & 7: Generate Dashboard and Final Report from WFV results
    logger.info("Generating final reports based on Walk-Forward Validation results...")
    try:
        config = pipeline.load_config()
        _execute_step("report", pipeline.run_report, config)
    except Exception as e:
        logger.error(f"Could not generate final report: {e}")

    logger.warning("--- Full (God-Mode) Pipeline Completed ---")


def release_gpu_resources(handle, use_gpu: bool) -> None:
    """Release NVML handle and log the result."""
    if use_gpu and "pynvml" in globals() and handle:
        try:
            pynvml.nvmlShutdown()
            logging.info("GPU resources released")
        except Exception as exc:  # pragma: no cover - unlikely NVML failure
            logging.warning(f"Failed to shut down NVML: {exc}")
    else:
        logging.info("GPU not available, running on CPU")


def run_mode(mode):
    """Run the selected mode."""
    if mode == "preprocess":
        run_preprocess()
    elif mode == "sweep":
        run_hyperparameter_sweep(DEFAULT_SWEEP_PARAMS)
    elif mode == "threshold":
        run_threshold_optimization()
    elif mode == "backtest":
        run_backtest()
    elif mode == "report":
        run_report()
    elif mode == "hyper_sweep":
        run_hyperparameter_sweep(DEFAULT_SWEEP_PARAMS)
    elif mode == "full_pipeline":
        run_full_pipeline()
    elif mode == "wfv":
        run_walkforward()  # [Patch] call simplified WFV runner
    elif mode == "all":
        # [Patch] Sweep then update config and run WFV
        run_hyperparameter_sweep(DEFAULT_SWEEP_PARAMS)
        # [Patch v6.3.1] Use os.path.join to support test monkeypatch for best_params
        candidates = [
            os.path.join(str(OUTPUT_DIR), "best_param.json"),
            os.path.join(str(OUTPUT_DIR), "best_params.json"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                with open(cand, "r", encoding="utf-8") as fh:
                    best_params = json.load(fh)
                update_config_from_dict(best_params)
                break
        run_walkforward()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def ensure_output_files(files):
    """[Patch v6.3.5] ตรวจสอบว่าไฟล์ผลลัพธ์จำเป็นมีครบ"""
    success = True
    for file in files:
        if not os.path.exists(file):
            logger.error("Required output file missing: %s", file)
            success = False
    return success


def load_features(path: str):
    """Load feature list from JSON file if it exists."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover - read error unlikely
        logger.error("Failed to load features from %s: %s", path, exc)
        return None


def save_features(features, path: str) -> None:
    """Persist feature list to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(features, fh, ensure_ascii=False, indent=2)


def generate_all_features(raw_data_paths: list[str]) -> list[str]:
    """Generate numeric feature names from the first CSV in the list."""
    if not raw_data_paths:
        return []
    path = raw_data_paths[0]
    if not os.path.exists(path):
        if FALLBACK_DIR:
            fallback_path = os.path.join(FALLBACK_DIR, os.path.basename(path))
            if os.path.exists(fallback_path):
                logger.warning(
                    "Raw data file not found: %s; using fallback %s",
                    path,
                    fallback_path,
                )
                path = fallback_path
            else:
                logger.warning(
                    "[Patch v6.4.6] Raw data file not found: %s. Proceeding with empty feature list.",
                    path,
                )
                return []
        else:
            logger.warning(
                "[Patch v6.4.6] Raw data file not found: %s. Proceeding with empty feature list.",
                path,
            )
            return []
    from src.utils.data_utils import safe_read_csv
    df_sample = safe_read_csv(path).head(500)
    features = [
        c
        for c in df_sample.columns
        if pd.api.types.is_numeric_dtype(df_sample[c])
        and c
        not in {
            "datetime",
            "is_tp",
            "is_sl",
            "Date",
            "Timestamp",
        }  # [Patch v6.7.2] skip date columns
        and c.lower() not in {"label", "target"}
    ]
    if len(features) < 10:
        logger.warning(
            "[Patch v6.9.22] Fewer than 10 numeric columns found (%d). Using DEFAULT_META_CLASSIFIER_FEATURES",
            len(features),
        )
        return DEFAULT_META_CLASSIFIER_FEATURES
    return features


def load_trade_log(
    filepath: str, min_rows: int = DEFAULT_TRADE_LOG_MIN_ROWS
) -> pd.DataFrame:
    """Load trade log and regenerate via backtest if rows are insufficient."""

    from src.trade_log_pipeline import load_or_generate_trade_log

    logger.info(f"[Patch v6.5.9] Attempting to load trade log from {filepath}")
    output_dir = getattr(config, "OUTPUT_DIR", Path(pipeline_config.output_dir))
    features_filename = pipeline_config.features_filename
    features_path = os.path.join(output_dir, features_filename)
    return load_or_generate_trade_log(
        filepath, min_rows=min_rows, features_path=features_path
    )


def ensure_trade_log(trade_log_path: str) -> str:
    """Ensure the required trade log exists, generating via walk-forward if missing."""
    if os.path.exists(trade_log_path):
        return trade_log_path
    alt = trade_log_path.replace(".csv.gz", ".csv")
    if os.path.exists(alt):
        return alt
    logger.warning(
        "[Patch v6.9.55] Trade log not found at %s, generating via walk-forward",
        trade_log_path,
    )
    try:
        run_walkforward()
    except Exception as exc:
        logger.error("Failed to generate trade log via WFV: %s", exc)
    if os.path.exists(trade_log_path):
        return trade_log_path
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(trade_log_path)


def main():
    """Main entry point for the ProjectP command-line interface."""
    print_logo()
    args = parse_args()

    if getattr(args, "menu", False):
        selected = interactive_menu()
        if not selected or selected == "exit":
            return
        if selected == "full_pipeline":
            run_full_pipeline()
            return
        args.mode = selected

    if args.auto_convert:
        src_dir = os.getenv("SOURCE_CSV_DIR", "data")
        dest_dir = os.getenv("DEST_CSV_DIR")
        out_base = dest_dir or setup_output_directory(os.getcwd(), "converted_csvs")
        auto_convert_csv(src_dir, output_path=os.path.join(out_base, "XAUUSD_M1_thai.csv"))
        sys.exit(0)

    if args.mode == "wfv" and args.all:
        print("\nINFO: Mode 'wfv --all' selected.")
        print("INFO: Starting the full Walk-Forward Validation (WFV) pipeline...")
        try:
            from wfv_orchestrator import main as run_wfv_pipeline
            run_wfv_pipeline()
        except ImportError:
            print(
                "WARNING: Could not import wfv_orchestrator directly. Running as a subprocess."
            )
            subprocess.run(["python", "wfv_orchestrator.py"], check=True)
        print("INFO: Full WFV pipeline has completed.")
        return

    if args.mode == "wfv" and not args.all:
        print("\nERROR: The 'wfv' mode requires the '--all' flag.")
        print("Usage: python ProjectP.py --mode wfv --all")
        sys.exit(1)

    run_mode(args.mode)
    return



def _script_main():
    args = parse_projectp_args()
    if getattr(args, "auto_convert", False):
        src_dir = os.getenv("SOURCE_CSV_DIR", "data")
        dest_dir = os.getenv("DEST_CSV_DIR")
        base = setup_output_directory(OUTPUT_BASE_DIR, OUTPUT_DIR_NAME)
        dest = dest_dir or os.path.join(base, "converted_csvs")
        os.makedirs(dest, exist_ok=True)
        auto_convert_csv(src_dir, output_path=dest)
        sys.exit(0)
    import main as pipeline_main
    # [Patch v6.9.43] return exit code instead of calling sys.exit for easier testing
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return pipeline_main.main()
    from src.utils.terminal_logger import terminal_logger
    term_log = os.path.join(OUTPUT_BASE_DIR, "terminal.log")
    with terminal_logger(term_log):
        return pipeline_main.main()


if __name__ == "__main__":
    _script_main()
