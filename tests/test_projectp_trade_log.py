import os
import ProjectP as proj


def test_ensure_trade_log_exists_no_generation(tmp_path, monkeypatch):
    log = tmp_path / "trade.csv"
    log.write_text("x")
    monkeypatch.setattr(proj, "run_walkforward", lambda: (_ for _ in ()).throw(AssertionError("should not run")))
    result = proj.ensure_trade_log(str(log))
    assert result == str(log)


def test_ensure_trade_log_generate(tmp_path, monkeypatch):
    log = tmp_path / "trade.csv"
    called = {}
    def fake_wfv():
        called['run'] = True
        log.write_text("ok")
    monkeypatch.setattr(proj, "run_walkforward", fake_wfv)
    result = proj.ensure_trade_log(str(log))
    assert called.get('run') is True
    assert os.path.exists(result)
