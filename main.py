import argparse
import subprocess
import os
import sys
import logging
import logging.config
import yaml
import pandas as pd
from src.data_loader import auto_convert_gold_csv

from src.utils.pipeline_config import (
    load_config,
    PipelineConfig,
    DEFAULT_CONFIG_FILE,
)
from src.utils.errors import PipelineError
from src.utils.hardware import has_gpu

logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    """Configure logging from YAML file."""
    with open("config/logger_config.yaml", "r") as f:
        log_cfg = yaml.safe_load(f.read())
    if level:
        log_cfg["root"]["level"] = level.upper()
    logging.config.dictConfig(log_cfg)


# [Patch v5.8.2] CLI pipeline orchestrator


def parse_args(args=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pipeline controller")
    parser.add_argument(
        "--mode",
        choices=["preprocess", "sweep", "threshold", "backtest", "report", "all"],
        default="all",
        help="Stage of the pipeline to run",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help="Path to pipeline configuration file",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level override (e.g., DEBUG)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (use limited rows for fast run)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Limit number of rows loaded from data (overrides debug default)",
    )
    parser.add_argument("--profile", action="store_true", help="Profile backtest stage")
    parser.add_argument(
        "--output-file",
        default="backtest_profile.prof",
        help="Profiling output file",
    )
    return parser.parse_args(args)





def run_preprocess(config: PipelineConfig, runner=subprocess.run) -> None:
    """Run data preprocessing stage."""
    logger.info("[Stage] preprocess")
    m1_path = config.raw_m1_filename
    auto_convert_gold_csv(os.path.dirname(m1_path), output_path=m1_path)
    try:
        runner([os.environ.get("PYTHON", "python"), "src/data_cleaner.py", m1_path], check=True)
        runner([os.environ.get("PYTHON", "python"), "ProjectP.py"], check=True)
    except subprocess.CalledProcessError as exc:
        logger.error("Preprocess failed", exc_info=True)
        raise PipelineError("preprocess stage failed") from exc


def run_sweep(config: PipelineConfig, runner=subprocess.run) -> None:
    """Run hyperparameter sweep stage."""
    logger.info("[Stage] sweep")
    try:
        runner(
            [os.environ.get("PYTHON", "python"), "tuning/hyperparameter_sweep.py"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Sweep failed", exc_info=True)
        raise PipelineError("sweep stage failed") from exc


def run_threshold(config: PipelineConfig, runner=subprocess.run) -> None:
    """Run threshold optimization stage."""
    logger.info("[Stage] threshold")
    try:
        runner(
            [os.environ.get("PYTHON", "python"), "threshold_optimization.py"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Threshold optimization failed", exc_info=True)
        raise PipelineError("threshold stage failed") from exc


def run_backtest_pipeline(features_df, price_df, model_path, threshold) -> None:
    """[Patch v5.9.12] Execute simple backtest pipeline."""
    logger.info("Running backtest with model=%s threshold=%s", model_path, threshold)
    try:
        from src.main import run_pipeline_stage
        run_pipeline_stage("backtest")
    except Exception:
        logger.exception("Internal backtest error")
        raise


def run_backtest(config: PipelineConfig, pipeline_func=run_backtest_pipeline) -> None:
    """Run backtest stage."""
    logger.info("[Stage] backtest")
    from config import config as cfg
    try:
        from ProjectP import load_trade_log
    except Exception:  # pragma: no cover - fallback for tests
        def load_trade_log(*_a, **_kw):
            return pd.DataFrame()

    trade_log_file = getattr(cfg, "TRADE_LOG_PATH", None)
    try:
        trade_df = load_trade_log(
            trade_log_file,
            min_rows=getattr(cfg, "MIN_TRADE_ROWS", 10),
        )
    except Exception as e:
        logger.error(f"Failed loading trade log: {e}")
        trade_df = pd.DataFrame(columns=["timestamp", "price", "signal"])
        logger.info("Initialized empty trade_df for pipeline execution.")
    else:
        logger.debug("Loaded trade log with %d rows", len(trade_df))
    model_dir = config.model_dir
    model_files = [
        f
        for f in os.listdir(model_dir)
        if f.startswith("model_") and f.endswith(".joblib")
    ]
    model_files.sort()
    model_path = os.path.join(model_dir, model_files[-1]) if model_files else None
    thresh_path = os.path.join(model_dir, config.threshold_file)
    threshold = None
    if os.path.exists(thresh_path):
        df = pd.read_csv(thresh_path)
        try:
            threshold_value = df["best_threshold"].median()
            # [Patch v6.6.7] Use Optuna best_threshold directly for backtest
            threshold = float(threshold_value) if not pd.isna(threshold_value) else None
        except KeyError:
            logging.warning(f"ไม่พบคอลัมน์ 'best_threshold' ในไฟล์ {thresh_path}")
            threshold = None
    try:
        pipeline_func(pd.DataFrame(), pd.DataFrame(), model_path, threshold)
    except Exception as exc:
        logger.error("Backtest failed", exc_info=True)
        raise PipelineError("backtest stage failed") from exc


def run_report(config: PipelineConfig) -> None:
    """Generate report stage."""
    logger.info("[Stage] report")
    try:
        from src.main import run_pipeline_stage
        run_pipeline_stage("report")
    except Exception as exc:
        logger.error("Report failed", exc_info=True)
        raise PipelineError("report stage failed") from exc


from src.pipeline_manager import PipelineManager


def run_all(config: PipelineConfig) -> None:
    """Run full pipeline via :class:`PipelineManager`."""
    logger.info("[Stage] all")
    PipelineManager(config).run_all()


def main(args=None) -> int:
    """Entry point for command-line execution."""
    parsed = parse_args(args)
    config = load_config(parsed.config)
    setup_logging(parsed.log_level or config.log_level)

    DEBUG_DEFAULT_ROWS = 2000
    if parsed.rows is not None:
        max_rows = parsed.rows
    elif parsed.debug:
        max_rows = DEBUG_DEFAULT_ROWS
    else:
        max_rows = None
    if max_rows:
        print(f"--- [DEBUG MODE] \u0e43\u0e0a\u0e49\u0e07\u0e32\u0e19 max_rows={max_rows} ---")
        import src.main as pipeline
        import src.strategy as strategy
        pipeline.sample_size = max_rows
        strategy.sample_size = max_rows
    else:
        print("--- [FULL PIPELINE] \u0e43\u0e0a\u0e49\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e17\u0e31\u0e49\u0e07\u0e2b\u0e21\u0e14 ---")

    if has_gpu():
        logger.info("GPU detected")
    else:
        logger.info("GPU not available, running on CPU")

    stage = parsed.mode
    logger.debug("Selected stage: %s", stage)

    try:
        if parsed.profile and stage == "backtest":
            import profile_backtest

            profile_backtest.run_profile(
                lambda: run_backtest(config), parsed.output_file
            )
            return 0
        if stage == "preprocess":
            run_preprocess(config)
        elif stage == "sweep":
            run_sweep(config)
        elif stage == "threshold":
            run_threshold(config)
        elif stage == "backtest":
            run_backtest(config)
        elif stage == "report":
            run_report(config)
        else:
            run_all(config)
    except PipelineError as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        return 1
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc, exc_info=True)
        return 1
    except ValueError as exc:
        logger.error("Invalid value: %s", exc, exc_info=True)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected errors
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
