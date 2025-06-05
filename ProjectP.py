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
        choices=["preprocess", "sweep", "threshold", "backtest", "report", "all"],
        default="preprocess",
        help="ขั้นตอนที่จะรัน",
    )
    return parser.parse_args(args)


def parse_args(args=None):  # backward compatibility
    return parse_projectp_args(args)


def run_preprocess():
    """รันขั้นตอนเตรียมข้อมูลและฝึกโมเดล."""
    return main()


def run_sweep():
    """รันการค้นหาค่าพารามิเตอร์."""
    # [Patch v5.7.2] Resolve sweep path relative to this file for Colab support
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_path = os.path.join(script_dir, "tuning", "hyperparameter_sweep.py")
    subprocess.run([sys.executable, sweep_path], check=True)


def run_threshold():
    """รันการปรับค่า threshold."""
    # [Patch v5.7.8] Resolve threshold script path relative to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    threshold_path = os.path.join(script_dir, "threshold_optimization.py")
    subprocess.run([sys.executable, threshold_path], check=True)


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


def run_report():
    """สร้างรายงานผลการทดสอบ."""
    pipeline.run_report()


def run_all_steps():
    """รันทุกโหมดต่อเนื่องกัน."""
    run_preprocess()
    run_sweep()
    run_threshold()
    run_backtest()
    run_report()


def run_mode(mode):
    """Run the selected mode."""
    if mode == "preprocess":
        run_preprocess()
    elif mode == "sweep":
        run_sweep()
    elif mode == "threshold":
        run_threshold()
    elif mode == "backtest":
        run_backtest()
    elif mode == "report":
        run_report()
    elif mode == "all":
        run_all_steps()
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    configure_logging()  # [Patch v5.5.14] Ensure consistent logging format
    args = parse_args()
    try:
        run_mode(args.mode)
        
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
