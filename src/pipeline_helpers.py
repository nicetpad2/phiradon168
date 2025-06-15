"""Pipeline-related helpers extracted from src.main"""
import logging
import os
import pandas as pd
from src.utils import load_settings, maybe_collect
from src.features import engineer_m1_features, save_features, load_features
from src.strategy import run_backtest_simulation_v34, plot_equity_curve

OUTPUT_DIR = ""
OPTUNA_N_TRIALS = 50
OPTUNA_DIRECTION = "maximize"
DATA_FILE_PATH_M1 = ""
INITIAL_CAPITAL = 100.0


def run_auto_threshold_stage():
    """Run Optuna-based threshold tuning if enabled."""
    from src.features import ENABLE_AUTO_THRESHOLD_TUNING
    if ENABLE_AUTO_THRESHOLD_TUNING:
        import threshold_optimization as topt
        logging.info("[Patch v6.2.4] Starting Auto Threshold Optimization")
        topt.run_threshold_optimization(
            output_dir=OUTPUT_DIR,
            trials=OPTUNA_N_TRIALS,
            study_name="threshold_wfv",
            direction=OPTUNA_DIRECTION,
            timeout=None,
        )
        logging.info("[Patch v6.2.4] Auto Threshold Optimization Completed")


def run_pipeline_stage(stage: str):
    from src import main as main_mod
    """Run a specific pipeline stage."""
    settings = load_settings()
    fmt = getattr(settings, "feature_format", "parquet")
    ext_map = {"parquet": ".parquet", "hdf5": ".h5"}
    ext = ext_map.get(fmt.lower(), ".csv")
    if stage == "preprocess":
        df = main_mod.load_validated_csv(DATA_FILE_PATH_M1, "M1")
        df = engineer_m1_features(df)
        out_path = os.path.join(OUTPUT_DIR, f"preprocessed{ext}")
        save_features(df, out_path, fmt)
        del df
        maybe_collect()
        logging.info("[Pipeline] Preprocess complete -> %s", out_path)
        return out_path
    if stage == "backtest":
        data_path = os.path.join(OUTPUT_DIR, f"preprocessed{ext}")
        if os.path.exists(data_path):
            df = load_features(data_path, fmt)
            if df is None:
                df = main_mod.load_validated_csv(DATA_FILE_PATH_M1, "M1")
        else:
            df = main_mod.load_validated_csv(DATA_FILE_PATH_M1, "M1")
        run_backtest_simulation_v34(df, label="WFV", initial_capital_segment=INITIAL_CAPITAL)
        logging.info("[Pipeline] Backtest completed")
        return None
    if stage == "report":
        metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
        if os.path.exists(metrics_path):
            pd.read_csv(metrics_path)
            plot_equity_curve([], "Equity", INITIAL_CAPITAL, OUTPUT_DIR, "report")
            logging.info("[Pipeline] Report generated")
        else:
            logging.warning("[Pipeline] No metrics to report")
        return None
    logging.error("Unknown stage: %s", stage)
    return None


def prepare_train_data():
    """Run PREPARE_TRAIN_DATA step programmatically."""
    logging.info("[Patch] Run Mode Selected: PREPARE_TRAIN_DATA (helper)")
    from src import main as main_mod
    return main_mod.main(run_mode="PREPARE_TRAIN_DATA")


def train_models():
    """Run TRAIN_MODEL_ONLY step programmatically."""
    logging.info("[Patch] Run Mode Selected: TRAIN_MODEL (helper)")
    from src import main as main_mod
    return main_mod.main(run_mode="TRAIN_MODEL_ONLY")
