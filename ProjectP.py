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

from src.utils.pipeline_config import load_config  # [Patch v6.7.17] dynamic config loader

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
from src.data_loader import auto_convert_gold_csv as auto_convert_csv

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
        choices=["preprocess", "sweep", "threshold", "backtest", "report", "all", "hyper_sweep", "wfv"],
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
    return main()


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
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".joblib")]
    model_files.sort()
    model_path = os.path.join(model_dir, model_files[-1]) if model_files else None
    thresh_path = os.path.join(model_dir, "threshold_wfv_optuna_results.csv")
    threshold = None
    if os.path.exists(thresh_path):
        df = pd.read_csv(thresh_path)
        if "best_threshold" in df.columns:
            threshold_val = df["best_threshold"].iloc[0]
            threshold = float(threshold_val) if not pd.isna(threshold_val) else None
        else:  # pragma: no cover - fallback for legacy column names
            threshold_val = df.median(numeric_only=True).mean()
            threshold = float(threshold_val) if not pd.isna(threshold_val) else None
    pipeline.run_backtest_pipeline(pd.DataFrame(), pd.DataFrame(), model_path, threshold)


def run_report() -> None:
    """สร้างรายงานผลการทดสอบ."""
    config = pipeline.load_config()
    pipeline.run_report(config)


def run_full_pipeline() -> None:
    """รันทุกโหมดต่อเนื่องกัน."""
    run_preprocess()
    # Step 2: Hyperparameter Sweep
    run_hyperparameter_sweep(DEFAULT_SWEEP_PARAMS)
    # Step 3: Auto-apply best hyperparameters from sweep
    summary_file = os.path.join(config.OUTPUT_DIR, 'hyperparameter_summary.csv')
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        if 'metric' in df.columns and df['metric'].notna().any():
            best = df.sort_values(by='metric', ascending=False).iloc[0]
            config.LEARNING_RATE = best.get('learning_rate', config.LEARNING_RATE)
            config.DEPTH = int(best.get('depth', config.DEPTH))
            config.L2_LEAF_REG = int(best.get('l2_leaf_reg', config.L2_LEAF_REG))
            logger.info(
                f"Applied best hyperparameters: lr={config.LEARNING_RATE}, "
                f"depth={config.DEPTH}, l2={config.L2_LEAF_REG}"
            )
        else:
            logger.warning(
                "ไม่มีคอลัมน์ metric ในไฟล์ sweep หรือไม่มีค่า metric ใช้ค่า default"
            )
    else:
        logger.warning(
            f"Hyperparameter summary not found at {summary_file}, using default parameters."
        )

    run_threshold_optimization()
    run_backtest()
    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics_summary_v32.csv")
    if os.path.exists(metrics_path):
        results_df = pd.read_csv(metrics_path)
    else:
        results_df = pd.DataFrame()
    generate_dashboard(results=results_df, output_filepath=os.path.join(config.OUTPUT_DIR, "dashboard.html"))
    run_report()


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
        df_sample = pd.read_csv(path, nrows=500)
    except FileNotFoundError:

        logger.warning(
            "[Patch v6.4.6] Raw data file not found: %s. Proceeding with empty feature list.",
            raw_data_paths[0],
        )
        # Proceed with no features if raw data is unavailable

        return []
    return [
        c
        for c in df_sample.columns
        if pd.api.types.is_numeric_dtype(df_sample[c])
        and c not in {"datetime", "is_tp", "is_sl", "Date", "Timestamp"}  # [Patch v6.7.2] skip date columns
        and c.lower() not in {"label", "target"}
    ]


def load_trade_log(filepath: str, min_rows: int = DEFAULT_TRADE_LOG_MIN_ROWS) -> pd.DataFrame:
    """Load trade log and regenerate via backtest if rows are insufficient."""

    from src.trade_log_pipeline import load_or_generate_trade_log

    logger.info(f"[Patch v6.5.9] Attempting to load trade log from {filepath}")
    output_dir = getattr(config, "OUTPUT_DIR", Path(pipeline_config.output_dir))
    features_filename = pipeline_config.features_filename
    features_path = os.path.join(output_dir, features_filename)
    return load_or_generate_trade_log(
        filepath, min_rows=min_rows, features_path=features_path
    )



if __name__ == "__main__":
    configure_logging()  # [Patch v5.5.14] Ensure consistent logging format
    args = parse_args(sys.argv[1:])
    if args.auto_convert:
        source_csv_dir = os.environ.get('SOURCE_CSV_DIR', '')
        dest_csv_dir = os.environ.get('DEST_CSV_DIR')
        if not dest_csv_dir:
            print("INFO: [AutoConvert] DEST_CSV_DIR not set, using default path.")
            main_output_dir = setup_output_directory(os.getcwd(), 'output_default')
            dest_csv_dir = os.path.join(main_output_dir, 'converted_csvs')
        os.makedirs(dest_csv_dir, exist_ok=True)
        if not source_csv_dir:
            print("\u2717 [AutoConvert] Error: SOURCE_CSV_DIR environment variable is not set.")
        else:
            auto_convert_csv(source_csv_dir, output_path=dest_csv_dir)
        sys.exit(0)
    DEBUG_DEFAULT_ROWS = 2000
    DEBUG_DEFAULT_ROWS = 2000
    if args.rows is not None:
        max_rows = args.rows
    elif args.debug:
        max_rows = DEBUG_DEFAULT_ROWS
    else:
        max_rows = None
    if max_rows:
        print(f"--- [DEBUG MODE] กำหนด max_rows={max_rows} ---")
        pipeline.sample_size = max_rows
        import src.strategy as strategy
        strategy.sample_size = max_rows
    else:
        print("--- [FULL PIPELINE] ใช้ข้อมูลทั้งหมด ---")
    # [Patch v6.3.5] ตรวจสอบไฟล์ผลลัพธ์ก่อนและหลังการทำงาน
    output_dir = getattr(config, "OUTPUT_DIR", Path(pipeline_config.output_dir))
    features_filename = pipeline_config.features_filename
    features_path = os.path.join(output_dir, features_filename)

    if not os.path.exists(features_path):
        logger.warning(
            f"features_main.json not found in {output_dir}; attempting to generate it."
        )
        try:
            from src import config as cfg
            os.makedirs(output_dir, exist_ok=True)
            feature_catalog = build_feature_catalog(
                data_dir=getattr(cfg, "DATA_DIR", output_dir),
                output_dir=output_dir,
            )
            with open(features_path, "w", encoding="utf-8") as fp:
                json.dump(feature_catalog, fp, ensure_ascii=False, indent=2)
            logger.info(
                "Generated features_main.json with %d entries.",
                len(feature_catalog),
            )
        except ImportError as ie:
            logger.error("Cannot import feature builder: %s. Aborting.", ie)
            sys.exit(1)
        except Exception as ex:
            logger.error("Failed to generate features_main.json: %s. Aborting.", ex)
            sys.exit(1)
    else:
        logger.info(
            "Loaded existing features_main.json (%d bytes)",
            os.path.getsize(features_path),
        )

    features_main = load_features(features_path)
    if features_main is None or len(features_main) < 10:
        logger.info("Generating fresh feature set...")
        raw_data_paths = [
            os.path.join(output_dir, "XAUUSD_M1.csv"),
            os.path.join(output_dir, "XAUUSD_M15.csv"),
        ]
        # [Patch v6.4.9] Fall back to default CSV paths from config
        if not os.path.exists(raw_data_paths[0]):
            try:
                from src import config as cfg
                raw_data_paths[0] = getattr(cfg, "DEFAULT_CSV_PATH_M1", raw_data_paths[0])
            except Exception:  # pragma: no cover - config import failure
                pass
        if len(raw_data_paths) > 1 and not os.path.exists(raw_data_paths[1]):
            try:
                from src import config as cfg
                raw_data_paths[1] = getattr(cfg, "DEFAULT_CSV_PATH_M15", raw_data_paths[1])
            except Exception:  # pragma: no cover - config import failure
                pass
        features_main = generate_all_features(raw_data_paths)
        save_features(features_main, features_path)
        logger.info("Feature set regenerated with %d rows", len(features_main))

    import glob
    # match both uncompressed (.csv) and gzip-compressed (.csv.gz) trade logs
    trade_pattern = os.path.join(output_dir, pipeline_config.trade_log_pattern)
    log_files = sorted(glob.glob(trade_pattern))
    if not log_files:

        logger.error(
            "[Patch v6.4.5] No trade_log CSV or CSV.GZ found in %s; aborting.",
            output_dir,
        )

        trade_pattern_gz = os.path.join(
            output_dir,
            pipeline_config.trade_log_pattern.replace(".csv*", ".csv.gz")
        )
        log_files = glob.glob(trade_pattern_gz)
    if not log_files and FALLBACK_DIR:
        fallback_pattern = os.path.join(
            FALLBACK_DIR, pipeline_config.trade_log_pattern
        )
        log_files = sorted(glob.glob(fallback_pattern))
        if log_files:
            logger.warning(
                "Trade log not found in %s; using fallback directory %s",
                output_dir,
                FALLBACK_DIR,
            )
    if not log_files:
        logger.warning(
            "[Patch v6.4.6] No trade_log CSV found in %s; initializing empty trade log.",
            output_dir,
        )
        # [Patch v6.7.2] Create a dummy trade log with placeholder rows (reduce to 9 rows)
        trade_log_file = os.path.join(output_dir, "trade_log_dummy.csv")
        dummy_cols = [
            "entry_time", "exit_time", "entry_price",
            "exit_price", "side", "profit",
        ]
        # populate 9 rows so that downstream sees insufficient data and trigger backtest
        dummy_rows = [
            {
                "entry_time": "",
                "exit_time": "",
                "entry_price": 0.0,
                "exit_price": 0.0,
                "side": "",
                "profit": 0.0,
            }
            for _ in range(9)
        ]
        pd.DataFrame(dummy_rows, columns=dummy_cols).to_csv(trade_log_file, index=False)
    else:
        log_files = sorted(log_files, key=lambda f: ("walkforward" not in f, f))
        trade_log_file = log_files[0]
    logger.info(
        "[Patch v5.8.15] Loaded trade log: %s", os.path.basename(trade_log_file)
    )


if __name__ == "__main__":
    configure_logging()
    from main import main as entry_main
    sys.exit(entry_main())
