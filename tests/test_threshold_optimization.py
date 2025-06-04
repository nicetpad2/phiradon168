import pandas as pd
import threshold_optimization as to


def test_run_threshold_optimization(tmp_path):
    df = to.run_threshold_optimization(str(tmp_path))
    assert df["median"].iloc[0] == 0.5
    assert (tmp_path / "threshold_wfv_optuna_results.csv").exists()
