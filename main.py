import argparse
import subprocess
import os
import pandas as pd

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
    print("[Stage] preprocess")
    subprocess.run([os.environ.get("PYTHON", "python"), "ProjectP.py"], check=True)


def run_sweep():
    print("[Stage] sweep")
    subprocess.run([os.environ.get("PYTHON", "python"), "tuning/hyperparameter_sweep.py"], check=True)


def run_threshold():
    print("[Stage] threshold")
    subprocess.run([os.environ.get("PYTHON", "python"), "threshold_optimization.py"], check=True)


def run_backtest_pipeline(features_df, price_df, model_path, threshold):
    """[Patch] Placeholder backtest pipeline"""
    print(f"Running backtest with model={model_path} threshold={threshold}")


def run_backtest():
    print("[Stage] backtest")
    model_dir = "models"
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".joblib")]
    model_files.sort()
    model_path = os.path.join(model_dir, model_files[-1]) if model_files else None
    thresh_path = os.path.join(model_dir, "threshold_wfv_optuna_results.csv")
    threshold = None
    if os.path.exists(thresh_path):
        df = pd.read_csv(thresh_path)
        if "median" in df.columns:
            threshold = float(df["median"].median())
    run_backtest_pipeline(pd.DataFrame(), pd.DataFrame(), model_path, threshold)


def run_report():
    print("[Stage] report")


def run_all():
    run_preprocess()
    run_sweep()
    run_threshold()
    run_backtest()
    run_report()


def main(args=None):
    parsed = parse_args(args)
    stage = parsed.stage
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
