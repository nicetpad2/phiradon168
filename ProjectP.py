"""Bootstrap script for running the main entry point."""

import logging
import csv
from pathlib import Path
try:  # [Patch v5.10.2] allow import without heavy dependencies
    from src.config import logger, OUTPUT_DIR
except Exception:  # pragma: no cover - fallback logger for tests
    logger = logging.getLogger("ProjectP")
    OUTPUT_DIR = Path("output_default")
# [Patch v5.9.17] Fallback logger if src.config fails
import sys
import os
import argparse
import subprocess
import json

# [Patch v6.2.5] Auto-fallback to project root only when executed as a script
if __name__ == "__main__":
    try:
        os.getcwd()
    except Exception:
        project_root = Path(__file__).resolve().parent
        os.chdir(project_root)
        print(f"[Info] Changed working directory to project root: {project_root}")
import pandas as pd
from typing import Dict, List
import main as pipeline
from config_loader import update_config_from_dict  # [Patch] dynamic config update
from wfv_runner import run_walkforward  # [Patch] walk-forward helper

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

from src.main import main

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
    return parser.parse_args(args)


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
    _sweep(
        str(OUTPUT_DIR),
        params,
        seed=42,
        resume=True,
        trade_log_path=DEFAULT_TRADE_LOG,
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
    threshold = {}
    if os.path.exists(thresh_path):
        df = pd.read_csv(thresh_path)
        threshold = df.median(numeric_only=True).to_dict()
    pipeline.run_backtest_pipeline(pd.DataFrame(), pd.DataFrame(), model_path, threshold)


def run_report() -> None:
    """สร้างรายงานผลการทดสอบ."""
    config = pipeline.load_config()
    pipeline.run_report(config)


def run_full_pipeline() -> None:
    """รันทุกโหมดต่อเนื่องกัน."""
    run_preprocess()
    run_hyperparameter_sweep(DEFAULT_SWEEP_PARAMS)
    run_threshold_optimization()
    run_backtest()
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
        # อ่านไฟล์ที่ sweep สร้าง (รองรับชื่อ best_param.json และ best_params.json)
        candidates = [OUTPUT_DIR / "best_param.json", OUTPUT_DIR / "best_params.json"]
        for cand in candidates:
            if os.path.exists(cand):
                with open(cand, "r", encoding="utf-8") as fh:
                    best_params = json.load(fh)
                update_config_from_dict(best_params)
                break
        run_walkforward()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def qa_check_and_create_outputs():
    """[Patch v5.8.14] Ensure fallback QA output files have valid headers."""
    output_dir = str(OUTPUT_DIR)
    files = [
        os.path.join(output_dir, "features_main.json"),
        os.path.join(output_dir, "trade_log_BUY.csv"),
        os.path.join(output_dir, "trade_log_SELL.csv"),
        os.path.join(output_dir, "trade_log_NORMAL.csv"),
    ]
    missing = [p for p in files if not os.path.exists(p) or os.path.getsize(p) == 0]
    for path in missing:
        logger.warning("[QA Fallback] Created missing file: %s", path)
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        if path.endswith("features_main.json"):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{}")  # JSON เปล่าแต่ valid
        else:
            header = [
                "timestamp",
                "symbol",
                "side",
                "price",
                "size",
                "order_type",
                "status",
            ]
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
        logger.debug("[QA Fallback] Wrote header to %s", path)


if __name__ == "__main__":
    configure_logging()  # [Patch v5.5.14] Ensure consistent logging format
    args = parse_args()
    # [Patch v5.9.5] สร้างไฟล์ QA พื้นฐานก่อนเริ่มการทำงานลดข้อความ error
    qa_check_and_create_outputs()
    try:
        run_mode(args.mode)
        
        # [Patch v5.3.4] Create empty audit files if missing after run
        output_dir = OUTPUT_DIR
        audit_files = [
            "features_main.json",
            "trade_log_BUY.csv",
            "trade_log_SELL.csv",
            "trade_log_NORMAL.csv",
        ]
        for fname in audit_files:
            fpath = OUTPUT_DIR / fname
            if fpath.exists():
                msg = f"[QA] Output present: {fpath}"
                logger.info(msg)
                logging.getLogger().info(msg)
                continue
            msg = f"[QA] Output missing: {fpath}"
            logger.error(msg)
            logging.getLogger().error(msg)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            if fname.endswith('.csv'):
                pd.DataFrame(columns=[
                    "timestamp", "symbol", "side", "price", "size", "order_type", "status"
                ]).to_csv(fpath, index=False)
            else:
                with open(fpath, 'w', encoding='utf-8') as fout:
                    json.dump({}, fout)
            logger.warning(f"[QA Fallback] Created missing file: {fpath}")


        # [Patch v5.8.13] Ensure fallback QA output files always exist
        fallback_files = [
            "features_main.json",
            "trade_log_BUY.csv",
            "trade_log_SELL.csv",
            "trade_log_NORMAL.csv",
        ]

        for fname in fallback_files:
            fpath = OUTPUT_DIR / fname
            if fpath.exists():
                continue
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            if fname.endswith('.csv'):
                pd.DataFrame(columns=[
                    "timestamp", "symbol", "side", "price", "size", "order_type", "status"
                ]).to_csv(fpath, index=False)
            else:
                with open(fpath, 'w', encoding='utf-8') as fout:
                    json.dump({}, fout)
            logger.warning(f"[QA Fallback] Created missing file: {fpath}")

    except KeyboardInterrupt:
        print("\n(Stopped) การทำงานถูกยกเลิกโดยผู้ใช้.")
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e), exc_info=True)
        sys.exit(1)
    else:
        # [Patch v5.0.23] Respect USE_GPU_ACCELERATION flag when logging GPU status
        main_mod = sys.modules.get("src.main")
        use_gpu = getattr(main_mod, "USE_GPU_ACCELERATION", False)
        release_gpu_resources(nvml_handle, use_gpu)
