import pandas as pd
import pytest
from src.wfv_monitor import walk_forward_validate
import src.wfv_monitor as wfv_monitor


def backtest_pass(train, test):
    return {"pnl": 100.0, "winrate": 0.7, "maxdd": 0.05, "auc": 0.8}


def backtest_fail(train, test):
    return {"pnl": 10.0, "winrate": 0.4, "maxdd": 0.2, "auc": 0.5}


def test_walk_forward_validate_pass():
    df = pd.DataFrame({"Close": range(20)}, index=pd.RangeIndex(20))
    kpi = {"profit": 50.0, "winrate": 0.6, "maxdd": 0.1, "auc": 0.6}
    res = walk_forward_validate(df, backtest_pass, kpi, n_splits=5)
    assert not res["failed"].any()


def test_walk_forward_validate_fail(monkeypatch):
    df = pd.DataFrame({"Close": range(20)}, index=pd.RangeIndex(20))
    kpi = {"profit": 50.0, "winrate": 0.6, "maxdd": 0.1, "auc": 0.6}
    called = {"n": 0}

    def dummy_retrain(fold, metrics):
        called["n"] += 1

    res = walk_forward_validate(df, backtest_fail, kpi, n_splits=5, retrain_func=dummy_retrain)
    assert res["failed"].all()
    assert called["n"] == 5


def test_unsorted_index_raises():
    df = pd.DataFrame({"Close": range(5)}, index=[5, 3, 4, 1, 2])
    with pytest.raises(ValueError):
        walk_forward_validate(df, backtest_pass, {"profit": 0}, n_splits=2)


def test_retrain_exception_logged(caplog):
    df = pd.DataFrame({"Close": range(10)}, index=pd.RangeIndex(10))
    kpi = {"profit": 50.0, "winrate": 0.6, "maxdd": 0.1, "auc": 0.6}

    def bad_retrain(fold, metrics):
        raise RuntimeError("boom")

    with caplog.at_level("ERROR", logger="src.wfv_monitor"):
        res = walk_forward_validate(
            df,
            backtest_fail,
            kpi,
            n_splits=3,
            retrain_func=bad_retrain,
        )
    assert len(res) == 3
    assert res["failed"].all()
    assert list(res["fold"]) == [0, 1, 2]


def test_threshold_boundary_pass():
    df = pd.DataFrame({"Close": range(10)}, index=pd.RangeIndex(10))
    metrics = {"pnl": 50.0, "winrate": 0.6, "maxdd": 0.1, "auc": 0.6}

    def bt(_, __):
        return metrics

    res = walk_forward_validate(df, bt, metrics, n_splits=2)
    assert not res["failed"].any()


def test_walk_forward_loop_basic(tmp_path):
    df = pd.DataFrame({"Close": range(10)}, index=pd.RangeIndex(10))
    kpi = {"profit": 0.0, "winrate": 0.5, "maxdd": 0.2, "auc": 0.5}

    out = tmp_path / "wfv.csv"

    res = wfv_monitor.walk_forward_loop(
        df,
        backtest_pass,
        kpi,
        train_window=3,
        test_window=2,
        step=2,
        output_path=str(out),
    )

    assert len(res) == 3
    assert out.exists()
    df_out = pd.read_csv(out)
    assert len(df_out) == 3


def test_walk_forward_loop_unsorted():
    df = pd.DataFrame({"Close": range(5)}, index=[2, 1, 3, 4, 0])
    with pytest.raises(ValueError):
        wfv_monitor.walk_forward_loop(df, backtest_pass, {"profit": 0}, 2, 1, 1)


def test_monitor_drift_warning(caplog):
    idx = pd.date_range('2024-01-01', periods=3, freq='D')
    train_df = pd.DataFrame({'feat': [1.0, 2.0, 3.0]}, index=idx)
    test_df = pd.DataFrame({'feat': [10.0, 11.0, 12.0]}, index=idx)
    with caplog.at_level('WARNING', logger='src.wfv_monitor'):
        res = wfv_monitor.monitor_drift(train_df, test_df, threshold=0.0)
    assert not res.empty
    assert res['drift'].any()


def test_monitor_drift_summary_warning(caplog):
    idx = pd.date_range('2024-01-01', periods=3, freq='D')
    train_df = pd.DataFrame({'feat': [1.0, 2.0, 3.0]}, index=idx)
    test_df = pd.DataFrame({'feat': [10.0, 11.0, 12.0]}, index=idx)
    with caplog.at_level('WARNING', logger='src.wfv_monitor'):
        res = wfv_monitor.monitor_drift_summary(train_df, test_df, threshold=0.0)
    assert not res.empty
    assert res['drift'].any()
