"""Bootstrap script for running the main entry point."""

from src.config import logger
import sys
import logging
import os
import argparse
import subprocess
import pandas as pd
import main as pipeline

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


def custom_helper_function():
    """Stubbed helper for tests."""
    return True


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["preprocess", "sweep", "threshold", "backtest", "report", "all"],
        default="preprocess",
        help="ขั้นตอนที่จะรัน",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.mode == "preprocess":
            suffix = main()
        elif args.mode == "sweep":
            subprocess.run([sys.executable, "tuning/hyperparameter_sweep.py"], check=True)
        elif args.mode == "threshold":
            subprocess.run([sys.executable, "threshold_optimization.py"], check=True)
        elif args.mode == "backtest":
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
        elif args.mode == "report":
            pipeline.run_report()
        else:  # all
            main()
            subprocess.run([sys.executable, "tuning/hyperparameter_sweep.py"], check=True)
            subprocess.run([sys.executable, "threshold_optimization.py"], check=True)
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
            pipeline.run_report()

        # [Patch v5.3.4] Create empty audit files if missing after run
        output_dir = "./output_default"
        audit_files = [
            "features_main.json",
            "trade_log_BUY.csv",
            "trade_log_SELL.csv",
            "trade_log_NORMAL.csv",
        ]
        for f in audit_files:
            fpath = os.path.join(output_dir, f)
            if os.path.exists(fpath):
                logger.info(f"[QA] Output present: {fpath}")
            else:
                logger.error(f"[QA] Output missing: {fpath}")
                os.makedirs(output_dir, exist_ok=True)
                open(fpath, "w", encoding="utf-8").close()
    except KeyboardInterrupt:
        print("\n(Stopped) การทำงานถูกยกเลิกโดยผู้ใช้.")
    except Exception as e:
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e), exc_info=True)
        sys.exit(1)
    else:
        # [Patch v5.0.23] Respect USE_GPU_ACCELERATION flag when logging GPU status
        main_mod = sys.modules.get("src.main")
        use_gpu = getattr(main_mod, "USE_GPU_ACCELERATION", False)
        if use_gpu and "pynvml" in globals() and nvml_handle:
            try:
                pynvml.nvmlShutdown()
                logging.info("GPU resources released")
            except Exception as e:
                logging.warning(f"Failed to shut down NVML: {e}")
        else:
            logging.info("GPU not available, running on CPU")
