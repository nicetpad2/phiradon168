import os
import sys
import pandas as pd
import logging
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.trade_logger import (
    export_trade_log,
    aggregate_trade_logs,
    Order,
    log_open_order,
    log_close_order,
    setup_trade_logger,
    print_qa_summary,
    save_trade_snapshot,
)


def test_export_trade_log_creates_file(tmp_path):
    df = pd.DataFrame({'a': [1]})
    out_dir = tmp_path / 'out'
    export_trade_log(df, str(out_dir), 'L1')
    assert (out_dir / 'trade_log_L1.csv').exists()
    # [Patch v5.9.2] Side-specific logs should be created even without 'side' column
    assert (out_dir / 'trade_log_BUY.csv').exists()
    assert (out_dir / 'trade_log_SELL.csv').exists()
    assert (out_dir / 'trade_log_NORMAL.csv').exists()
    qa_dir = out_dir / 'qa_logs'
    assert not (qa_dir / 'L1_trade_qa.log').exists()
    summary = qa_dir / 'qa_summary_L1.log'
    assert summary.exists()
    assert summary.read_text() == f"Trade Log QA: {len(df)} trades, saved {out_dir / 'trade_log_L1.csv'}\n"


def test_export_trade_log_empty_creates_audit(tmp_path):
    out_dir = tmp_path / 'out2'
    export_trade_log(pd.DataFrame(), str(out_dir), 'L2')
    log_file = out_dir / 'trade_log_L2.csv'
    qa_file = out_dir / 'qa_logs' / 'L2_trade_qa.log'
    relax_file = out_dir / 'qa_logs' / 'relax_threshold_L2.log'
    # Side logs should still be created as empty files
    assert (out_dir / 'trade_log_BUY.csv').exists()
    assert (out_dir / 'trade_log_SELL.csv').exists()
    assert (out_dir / 'trade_log_NORMAL.csv').exists()
    assert log_file.exists()
    assert qa_file.exists()
    assert qa_file.read_text() == "[QA] No trade. Output file generated as EMPTY.\n"
    assert relax_file.exists()


def test_aggregate_trade_logs(tmp_path):
    dir1 = tmp_path / 'f1'
    dir2 = tmp_path / 'f2'
    df1 = pd.DataFrame({'a': [1]})
    df2 = pd.DataFrame({'a': [2]})
    export_trade_log(df1, str(dir1), 'BUY')
    export_trade_log(df2, str(dir2), 'BUY')
    out_file = tmp_path / 'combined' / 'trade_log_BUY.csv'
    aggregate_trade_logs([str(dir1), str(dir2)], str(out_file), 'BUY')
    combined = pd.read_csv(out_file)
    assert len(combined) == 2
    qa_log = out_file.parent / 'trade_log_BUY_qa.log'
    assert qa_log.exists()


def test_order_dataclass_logging(caplog):
    logger = logging.getLogger('test_trade_logger')
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger='test_trade_logger'):
        order = Order(
            side='BUY',
            entry_price=1.0,
            sl_price=0.99,
            open_time=pd.Timestamp('2025-01-01 00:00', tz='Asia/Bangkok'),
        )
        log_open_order(order, logger)
        log_close_order(
            order,
            exit_price=1.01,
            reason='take profit',
            close_time=pd.Timestamp('2025-01-01 01:00', tz='Asia/Bangkok'),
            trade_log=logger,
        )
    assert 'Order Opened' in caplog.text
    assert 'take profit' in caplog.text
    assert '2024-12-31T17:00:00+00:00' in caplog.text


def test_setup_trade_logger(tmp_path):
    log_path = tmp_path / 't.log'
    trade_log = setup_trade_logger(str(log_path), max_bytes=100, backup_count=1)
    assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in trade_log.handlers)


def test_print_qa_summary(tmp_path, caplog):
    qa_dir = tmp_path / 'qa_logs'
    qa_dir.mkdir(parents=True)
    summary_file = qa_dir / 'qa_summary_X.log'
    summary_file.write_text('OK', encoding='utf-8')
    with caplog.at_level(logging.INFO):
        text = print_qa_summary(str(tmp_path))
    assert 'OK' in caplog.text
    assert text.strip() == 'OK'


def test_print_qa_summary_missing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        text = print_qa_summary(str(tmp_path))
    assert text == ''
    assert '[QA-WARNING]' in caplog.text


def test_save_trade_snapshot(tmp_path):
    out = tmp_path / 'snap.csv'
    data = {'time': '2025-01-01', 'price': 1.0, 'atr': 0.2, 'result': 5}
    save_trade_snapshot(data, str(out))
    assert out.exists()
    df = pd.read_csv(out)
    assert df.iloc[0]['price'] == 1.0


def test_order_as_dict():
    ts = pd.Timestamp('2025-01-01 00:00:00')
    order = Order(side='SELL', entry_price=1.5, sl_price=1.0, open_time=ts)
    assert order.as_dict() == {
        'side': 'SELL',
        'entry_price': 1.5,
        'sl_price': 1.0,
        'open_time': ts,
    }


def test_export_trade_log_with_splitter(tmp_path, monkeypatch):
    df = pd.DataFrame({'side': ['BUY'], 'price': [1]})
    called = {}

    def fake_split(trades, output_dir):
        called['args'] = (trades, output_dir)

    monkeypatch.setattr('src.utils.trade_splitter.split_trade_log', fake_split)
    monkeypatch.setattr('src.utils.trade_splitter.has_buy_sell', lambda x: True)

    out_dir = tmp_path / 'out_split'
    export_trade_log(df, str(out_dir), 'L3', fund_name='FundX')
    assert called['args'][0].equals(df)
    assert called['args'][1] == str(out_dir)
    assert (out_dir / 'qa_logs' / 'FundX' / 'qa_summary_L3.log').exists()


def test_ensure_ts_naive():
    ts = pd.Timestamp('2025-01-01 00:00:00')
    from src.utils.trade_logger import _ensure_ts
    result = _ensure_ts(ts)
    assert str(result.tz) == 'UTC'


def test_log_close_order_stop_loss_warning(caplog):
    logger = logging.getLogger('warn_logger')
    logger.setLevel(logging.INFO)
    order = Order(
        side='BUY', entry_price=1.0, sl_price=0.9, open_time=pd.Timestamp('2025-01-01', tz='UTC')
    )
    with caplog.at_level(logging.WARNING, logger='warn_logger'):
        log_close_order(
            order,
            exit_price=0.9,
            reason='Hit SL',
            close_time=pd.Timestamp('2025-01-01 01:00', tz='UTC'),
            trade_log=logger,
        )
    assert 'Hit SL' in caplog.text
    assert 'WARNING' in caplog.text


def test_save_trade_snapshot_type_error(tmp_path):
    with pytest.raises(TypeError):
        save_trade_snapshot(['bad'], str(tmp_path / 'snap.csv'))


def test_print_qa_summary_error(tmp_path, caplog):
    qa_dir = tmp_path / 'qa_logs'
    qa_dir.mkdir()
    bad = qa_dir / 'qa_summary_bad.log'
    bad.mkdir()  # make directory so open() fails
    with caplog.at_level(logging.ERROR):
        text = print_qa_summary(str(tmp_path))
    assert text == ''
    assert 'Failed reading' in caplog.text
