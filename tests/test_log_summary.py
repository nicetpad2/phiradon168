import pandas as pd
from src.log_analysis import summarize_trade_log


def test_summarize_trade_log(tmp_path):
    log = tmp_path / 'log.txt'
    log.write_text(
        'INFO:root:   Attempting to Open New Order for SELL at 2023-01-01 10:00:00+00:00\n'
        'INFO:root:      Order Closing: Time=2023-01-01 10:10:00+00:00, Final Reason=SL, ExitPrice=1900, EntryTime=2023-01-01 10:00:00+00:00\n'
        'INFO:root:         [Patch PnL Final] Closed Lot=0.01, PnL(Net USD)=-1.0\n'
    )
    summary = summarize_trade_log(str(log))
    assert 'hourly' in summary
    assert 'equity_curve' in summary

