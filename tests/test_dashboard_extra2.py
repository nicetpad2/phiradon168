import pandas as pd
import pytest
from src.dashboard import create_dashboard, plot_wfv_summary


def test_create_dashboard_with_shap(tmp_path):
    equity = pd.Series([1, 2], index=pd.date_range('2024-01-01', periods=2))
    dd = pd.Series([0, -1], index=equity.index)
    ret = pd.Series([0.1, -0.2])
    image = tmp_path / 's.png'
    image.write_bytes(b'data')
    fig = create_dashboard(equity, dd, ret, str(image))
    assert fig.layout.images


def test_plot_wfv_summary_error():
    with pytest.raises(ValueError):
        plot_wfv_summary(pd.DataFrame())
