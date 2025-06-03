import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.trade_logger import export_trade_log


def test_export_trade_log_creates_file(tmp_path):
    df = pd.DataFrame({'a': [1]})
    out_dir = tmp_path / 'out'
    export_trade_log(df, str(out_dir), 'L1')
    assert (out_dir / 'trade_log_L1.csv').exists()
    assert not (out_dir / 'L1_trade_qa.log').exists()


def test_export_trade_log_empty_creates_audit(tmp_path):
    out_dir = tmp_path / 'out2'
    export_trade_log(pd.DataFrame(), str(out_dir), 'L2')
    log_file = out_dir / 'trade_log_L2.csv'
    qa_file = out_dir / 'L2_trade_qa.log'
    assert log_file.exists()
    assert qa_file.exists()
    assert qa_file.read_text() == "[QA] No trade. Output file generated as EMPTY.\n"
