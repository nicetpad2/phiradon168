import threshold_optimization as to


def test_parse_args_defaults():
    args = to.parse_args([])
    assert args.trials == 10
    assert args.study_name == "threshold-wfv"
    assert args.direction == "maximize"
    assert args.output_dir == "models"


def test_run_threshold_optimization(tmp_path):
    df = to.run_threshold_optimization(
        output_dir=str(tmp_path), trials=1, timeout=1
    )
    assert set(df.columns) == {"best_threshold", "best_value"}
    assert 0.0 <= df["best_threshold"].iloc[0] <= 1.0
    assert (tmp_path / "threshold_wfv_optuna_results.csv").exists()
    assert (tmp_path / "threshold_wfv_optuna_results.json").exists()
