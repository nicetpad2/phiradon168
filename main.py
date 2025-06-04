import argparse
import subprocess
import os
import logging
import pandas as pd

from src.utils.pipeline_config import load_config, PipelineConfig
from src.utils.errors import PipelineError
from src.utils.hardware import has_gpu

config: PipelineConfig = load_config()
logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

if has_gpu():
    logger.info("GPU detected")
else:
    logger.info("GPU not available, running on CPU")

# [Patch v5.5.9] CLI pipeline orchestrator

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["preprocess", "sweep", "threshold", "backtest", "report", "all"],
        default="all",
    )
    parser.add_argument("--profile", action="store_true", help="Profile backtest stage")
    parser.add_argument("--output-file", default="backtest_profile.prof", help="Profiling output file")
    return parser.parse_args(args)


def run_preprocess():
    logger.info("[Stage] preprocess")
    try:
        subprocess.run([os.environ.get("PYTHON", "python"), "ProjectP.py"], check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Preprocess failed")
        raise PipelineError("preprocess stage failed") from exc


def run_sweep():
    logger.info("[Stage] sweep")
    try:
        subprocess.run([os.environ.get("PYTHON", "python"), "tuning/hyperparameter_sweep.py"], check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Sweep failed")
        raise PipelineError("sweep stage failed") from exc


def run_threshold():
    logger.info("[Stage] threshold")
    try:
        subprocess.run([os.environ.get("PYTHON", "python"), "threshold_optimization.py"], check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Threshold optimization failed")
        raise PipelineError("threshold stage failed") from exc


def run_backtest_pipeline(features_df, price_df, model_path, threshold):
    """[Patch] Placeholder backtest pipeline"""
    logger.info("Running backtest with model=%s threshold=%s", model_path, threshold)


def run_backtest():
    logger.info("[Stage] backtest")
    model_dir = config.model_dir
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".joblib")]
    model_files.sort()
    model_path = os.path.join(model_dir, model_files[-1]) if model_files else None
    thresh_path = os.path.join(model_dir, config.threshold_file)
    threshold = None
    if os.path.exists(thresh_path):
        df = pd.read_csv(thresh_path)
        if "median" in df.columns:
            threshold = float(df["median"].median())
    try:
        run_backtest_pipeline(pd.DataFrame(), pd.DataFrame(), model_path, threshold)
    except Exception as exc:
        logger.error("Backtest failed")
        raise PipelineError("backtest stage failed") from exc


def run_report():
    logger.info("[Stage] report")


def run_all():
    logger.info("[Stage] all")
    run_preprocess()
    run_sweep()
    run_threshold()
    run_backtest()
    run_report()


def main(args=None):
    parsed = parse_args(args)
    stage = parsed.stage
    logger.debug("Selected stage: %s", stage)
    if parsed.profile and stage == "backtest":
        import profile_backtest

        profile_backtest.run_profile(run_backtest, parsed.output_file)
        return
    if stage == "preprocess":
        run_preprocess()
    elif stage == "sweep":
        run_sweep()
    elif stage == "threshold":
        run_threshold()
    elif stage == "backtest":
        run_backtest()
    elif stage == "report":
        run_report()
    else:
        run_all()


if __name__ == "__main__":
    main()
