import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
from src.param_stability import analyze_param_stability, save_fold_params



def test_analyze_param_stability_unstable(tmp_path):
    params = [
        {"tp": 10, "sl": 7, "atr": 1.5},
        {"tp": 30, "sl": 21, "atr": 4.5},
        {"tp": 5, "sl": 3, "atr": 1.0},
    ]
    df = analyze_param_stability(params, threshold=0.3)
    assert df[df["param"] == "tp"]["unstable"].iloc[0]
    out = tmp_path / "params.csv"
    save_fold_params(params, str(out))
    assert out.is_file()


def test_analyze_param_stability_stable():
    params = [
        {"tp": 10, "sl": 7, "atr": 1.5},
        {"tp": 11, "sl": 7.2, "atr": 1.6},
        {"tp": 9.5, "sl": 6.8, "atr": 1.4},
    ]
    df = analyze_param_stability(params, threshold=0.3)
    assert not df["unstable"].any()

