import pandas as pd
from reporting.dashboard import plot_underwater_curve
import pytest


def test_plot_underwater_curve():
    eq = pd.Series([1, 0.9, 1.1], index=pd.date_range('2024-01-01', periods=3))
    fig = plot_underwater_curve(eq)
    assert fig.data


def test_plot_underwater_curve_error():
    with pytest.raises(ValueError):
        plot_underwater_curve(pd.Series(dtype=float))
