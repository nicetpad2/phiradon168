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
from main import (
    run_preprocess as pipeline_run_preprocess,
)  # [Patch v6.9.32] use pipeline preprocess

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

from src.main import main, setup_output_directory
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


def custom_helper_function():
    """Stubbed helper for tests."""
    return True


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


def run_hyperparameter_sweep(params: Dict[str, List[float]]) -> None:
    """รันการค้นหาค่าพารามิเตอร์."""
    logger.debug(f"Starting sweep with params: {params}")
    from tuning.hyperparameter_sweep import run_sweep as _sweep, DEFAULT_TRADE_LOG

    m1_path = os.path.join(str(OUTPUT_DIR), "final_data_m1_v32_walkforward.csv.gz")
    logger.info("[Patch v6.6.6] Running sweep with m1_path=%s", m1_path)

    _sweep(
        str(OUTPUT_DIR),
        params,
        seed=42,
        resume=True,
        trade_log_path=DEFAULT_TRADE_LOG,
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
    """รันทุกโหมดต่อเนื่องกัน."""
    _execute_step("preprocess", run_preprocess)
    # Step 2: Hyperparameter Sweep
    _execute_step("sweep", run_hyperparameter_sweep, DEFAULT_SWEEP_PARAMS)
    # Step 3: Auto-apply best hyperparameters from sweep
    summary_file = os.path.join(config.OUTPUT_DIR, "hyperparameter_summary.csv")
    if os.path.exists(summary_file):
        from src.utils.data_utils import safe_read_csv

        df = safe_read_csv(summary_file)
        if "metric" in df.columns and df["metric"].notna().any():
            best = df.sort_values(by="metric", ascending=False).iloc[0]
            config.LEARNING_RATE = best.get("learning_rate", config.LEARNING_RATE)
            config.DEPTH = int(best.get("depth", config.DEPTH))
            config.L2_LEAF_REG = int(best.get("l2_leaf_reg", config.L2_LEAF_REG))
            logger.info(
                f"Applied best hyperparameters: lr={config.LEARNING_RATE}, "
                f"depth={config.DEPTH}, l2={config.L2_LEAF_REG}"
            )
        else:
            logger.warning("ไม่มีคอลัมน์ metric ในไฟล์ sweep หรือไม่มีค่า metric ใช้ค่า default")
    else:
        logger.warning(
            f"Hyperparameter summary not found at {summary_file}, using default parameters."
        )

    _execute_step("threshold", run_threshold_optimization)
    _execute_step("backtest", run_backtest)
    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics_summary_v32.csv")
    if os.path.exists(metrics_path):
        from src.utils.data_utils import safe_read_csv

        results_df = safe_read_csv(metrics_path)
    else:
        results_df = pd.DataFrame()
    _execute_step(
        "dashboard",
        generate_dashboard,
        results=results_df,
        output_filepath=os.path.join(config.OUTPUT_DIR, "dashboard.html"),
    )
    _execute_step("report", run_report)
    ensure_output_files([metrics_path])


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
    """[Patch v6.3.5] ตรวจสอบว่าไฟล์ผลลัพธ์จำเป็นมีครบ หากขาดให้หยุดการทำงาน"""
    for file in files:
        if not os.path.exists(file):
            logger.error("Required output file missing: %s. Aborting.", file)
            # Abort to prevent downstream dummy placeholders
            sys.exit(1)


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
    if not os.path.exists(path) and FALLBACK_DIR:
        fallback_path = os.path.join(FALLBACK_DIR, os.path.basename(path))
        if os.path.exists(fallback_path):
            logger.warning(
                "Raw data file not found: %s; using fallback %s",
                path,
                fallback_path,
            )
            path = fallback_path
    try:
        from src.utils.data_utils import safe_read_csv

        df_sample = safe_read_csv(path).head(500)
    except FileNotFoundError:

        logger.warning(
            "[Patch v6.4.6] Raw data file not found: %s. Proceeding with empty feature list.",
            raw_data_paths[0],
        )
        # Proceed with no features if raw data is unavailable

        return []
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


def main():
    """Main entry point for the ProjectP command-line interface."""
    parser = argparse.ArgumentParser(
        description="ProjectP Pipeline Manager.",
        usage="python ProjectP.py <mode> [--all] [--step-name <step>]",
    )

    parser.add_argument(
        "mode",
        nargs="?",
        help=(
            "The mode to run. Available modes: 'full_pipeline', 'wfv', "
            "'tune', 'backtest', 'step'."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="A flag to run the full process. Primarily used with 'wfv' mode.",
    )
    parser.add_argument(
        "--step-name",
        type=str,
        help="Specify the step name from pipeline.yaml. Used with 'step' mode.",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.mode == "wfv":
        if args.all:
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
        else:
            print("\nERROR: The 'wfv' mode requires the '--all' flag.")
            print("Usage: python ProjectP.py wfv --all")
            sys.exit(1)

    elif args.mode == "full_pipeline":
        run_full_pipeline()

    elif args.mode == "step":
        if args.step_name:
            print(f"\nINFO: Mode 'step' selected. Executing step: '{args.step_name}'...")
            print(f"INFO: Logic for step '{args.step_name}' should be implemented here.")
        else:
            print("\nERROR: The 'step' mode requires the '--step-name' argument.")
            print("Usage: python ProjectP.py step --step-name <name_of_your_step>")
            sys.exit(1)

    else:
        print(
            f"\nERROR: Invalid or incomplete command. Mode '{args.mode}' is not recognized or is missing arguments."
        )
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
