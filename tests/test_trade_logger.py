import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.trade_logger import export_trade_log, aggregate_trade_logs


def test_export_trade_log_creates_file(tmp_path):
    df = pd.DataFrame({'a': [1]})
    out_dir = tmp_path / 'out'
    export_trade_log(df, str(out_dir), 'L1')
    assert (out_dir / 'trade_log_L1.csv').exists()
    assert not (out_dir / 'L1_trade_qa.log').exists()
    summary = out_dir / 'qa_summary_L1.log'
    assert summary.exists()
    assert summary.read_text() == f"Trade Log QA: {len(df)} trades, saved {out_dir / 'trade_log_L1.csv'}\n"


def test_export_trade_log_empty_creates_audit(tmp_path):
    out_dir = tmp_path / 'out2'
    export_trade_log(pd.DataFrame(), str(out_dir), 'L2')
    log_file = out_dir / 'trade_log_L2.csv'
    qa_file = out_dir / 'L2_trade_qa.log'
    assert log_file.exists()
    assert qa_file.exists()
    assert qa_file.read_text() == "[QA] No trade. Output file generated as EMPTY.\n"


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
