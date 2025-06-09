import pandas as pd
from src.log_analysis import calculate_equity_curve, calculate_expectancy_by_period, plot_equity_curve


def test_calculate_equity_curve_basic():
    df = pd.DataFrame({'PnL': [1.0, -0.5, 2.0]})
    curve = calculate_equity_curve(df)
    assert list(curve) == [1.0, 0.5, 2.5]


def test_calculate_expectancy_by_period_hourly():
    data = {
        'EntryTime': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:30', '2023-01-01 11:00']),
        'PnL': [1.0, -0.5, 2.0],
    }
    df = pd.DataFrame(data)
    exp = calculate_expectancy_by_period(df)
    assert '2023-01-01 10' in exp.index.astype(str)[0]
    assert len(exp) == 2


def test_plot_equity_curve():
    curve = pd.Series([0, 1, 0.5])
    fig = plot_equity_curve(curve)
    assert hasattr(fig, 'savefig')
