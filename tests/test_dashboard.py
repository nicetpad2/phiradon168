import pandas as pd
from src.dashboard import create_dashboard, save_dashboard


def test_create_dashboard(tmp_path):
    equity = pd.Series([100, 102, 105], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    dd = pd.Series([0, -0.02, -0.05], index=equity.index)
    ret = pd.Series([0.01, -0.02, 0.03])
    fig = create_dashboard(equity, dd, ret)
    out_file = tmp_path / "dash.html"
    save_dashboard(fig, str(out_file))
    assert out_file.exists()
    text = out_file.read_text()
    assert "Equity Curve" in text
