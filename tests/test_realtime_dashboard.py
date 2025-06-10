import pandas as pd
import pytest
from src.realtime_dashboard import (
    compute_drawdown,
    compute_equity_curve,
    check_drawdown_alert,
)
import src.realtime_dashboard as realtime_dashboard


def test_compute_equity_curve():
    df = pd.DataFrame({'pnl': [1, -0.5, 2]})
    equity = compute_equity_curve(df)
    assert list(equity) == [1, 0.5, 2.5]


def test_compute_drawdown():
    equity = pd.Series([1, 2, 1.5])
    dd = compute_drawdown(equity)
    assert pytest.approx(dd.tolist()) == [0.0, 0.0, -0.25]


def test_check_drawdown_alert():
    dd = pd.Series([0.0, -0.03, -0.06])
    assert check_drawdown_alert(dd, threshold=0.05) is True
    assert check_drawdown_alert(pd.Series(), threshold=0.05) is False


def test_load_trade_log(tmp_path):
    df = pd.DataFrame({
        'entry_time': pd.date_range('2020-01-01', periods=2, freq='D'),
        'exit_time': pd.date_range('2020-01-02', periods=2, freq='D'),
        'pnl': [1.0, -2.0]
    })
    csv_path = tmp_path / "log.csv"
    df.to_csv(csv_path, index=False)
    loaded = realtime_dashboard.load_trade_log(str(csv_path))
    pd.testing.assert_frame_equal(loaded, df)


def test_load_trade_log_auto_generate(tmp_path, monkeypatch):
    df_res = pd.DataFrame({"pnl": [1.0, -1.0]})
    monkeypatch.setattr(realtime_dashboard.wfv_runner, "run_walkforward", lambda nrows=20: df_res)
    path = tmp_path / "missing.csv"
    loaded = realtime_dashboard.load_trade_log(str(path))
    assert path.exists()
    assert "pnl" in loaded.columns
    df = pd.DataFrame({'entry_time': ['2020-01-01'], 'exit_time': ['2020-01-01']})
    bad_path = tmp_path / "bad.csv"
    df.to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        realtime_dashboard.load_trade_log(str(bad_path))


def test_compute_equity_drawdown_empty():
    assert realtime_dashboard.compute_equity_curve(pd.DataFrame({'pnl': []})).empty
    assert realtime_dashboard.compute_drawdown(pd.Series(dtype=float)).empty


def test_generate_dashboard(monkeypatch, tmp_path):
    df = pd.DataFrame({
        'entry_time': pd.date_range('2020-01-01', periods=2, freq='D'),
        'exit_time': pd.date_range('2020-01-02', periods=2, freq='D'),
        'pnl': [1.0, -2.0]
    })
    log_path = tmp_path / 'log.csv'
    df.to_csv(log_path, index=False)

    dummy_fig = object()
    def fake_dashboard(equity, dd, returns):
        return dummy_fig
    monkeypatch.setattr(realtime_dashboard, 'create_dashboard', fake_dashboard)

    fig, alert = realtime_dashboard.generate_dashboard(str(log_path), threshold=0.5)
    assert fig is dummy_fig
    assert alert is True


def test_run_streamlit_dashboard_import_error(monkeypatch):
    monkeypatch.setattr(realtime_dashboard, 'st', None)
    with pytest.raises(ImportError):
        realtime_dashboard.run_streamlit_dashboard('path')


def test_run_streamlit_dashboard(monkeypatch):
    class DummyPlaceholder:
        def __init__(self):
            self.plotted = False
        def plotly_chart(self, fig, use_container_width=True):
            self.plotted = True
    class DummyStreamlit:
        def __init__(self):
            self.set_page_config_called = False
            self.errors = []
            self.placeholder = DummyPlaceholder()
        def set_page_config(self, page_title):
            self.set_page_config_called = True
        def empty(self):
            return self.placeholder
        def error(self, msg):
            self.errors.append(msg)
    dummy_st = DummyStreamlit()

    def fake_generate(path, threshold):
        return 'fig', True
    def fake_sleep(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(realtime_dashboard, 'st', dummy_st)
    monkeypatch.setattr(realtime_dashboard, 'generate_dashboard', fake_generate)
    monkeypatch.setattr(realtime_dashboard.time, 'sleep', fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        realtime_dashboard.run_streamlit_dashboard('log', refresh_sec=0)

    assert dummy_st.set_page_config_called
    assert dummy_st.placeholder.plotted
    assert dummy_st.errors == ['Drawdown exceeds 5.0%!']
