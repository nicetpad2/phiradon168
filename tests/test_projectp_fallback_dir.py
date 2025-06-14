import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


def test_projectp_fallback_dir(monkeypatch, tmp_path):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    # create raw data files
    df = pd.DataFrame({"A": [1], "B": [2]})
    df.to_csv(fallback / "XAUUSD_M1.csv", index=False)
    df.to_csv(fallback / "XAUUSD_M15.csv", index=False)
    # create minimal trade log
    pd.DataFrame({"profit": [1.0]}).to_csv(
        fallback / "trade_log_v32_walkforward.csv",
        index=False,
    )

    monkeypatch.setenv("PROJECTP_FALLBACK_DIR", str(fallback))
    import importlib
    import ProjectP
    importlib.reload(ProjectP)
    feats = ProjectP.generate_all_features([str(tmp_path / "XAUUSD_M1.csv")])
    assert feats == ProjectP.DEFAULT_META_CLASSIFIER_FEATURES
