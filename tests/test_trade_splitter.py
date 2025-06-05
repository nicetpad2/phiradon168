import pandas as pd
import logging
from src.utils.trade_splitter import split_trade_log, has_buy_sell


def test_split_trade_log(tmp_path, caplog):
    df = pd.DataFrame({
        'side': ['BUY', 'SELL', 'B', 's', '1', '0', 'HOLD'],
        'entry_price': [1,1,1,1,1,1,1],
    })
    out_dir = tmp_path / 'out'
    with caplog.at_level(logging.INFO):
        split_trade_log(df, str(out_dir))
    assert (out_dir / 'trade_log_BUY.csv').exists()
    assert (out_dir / 'trade_log_SELL.csv').exists()
    assert (out_dir / 'trade_log_NORMAL.csv').exists()
    buy = pd.read_csv(out_dir / 'trade_log_BUY.csv')
    sell = pd.read_csv(out_dir / 'trade_log_SELL.csv')
    normal = pd.read_csv(out_dir / 'trade_log_NORMAL.csv')
    assert len(buy) == 3  # BUY,B,1
    assert len(sell) == 3  # SELL,s or 0
    assert len(normal) == 1  # others
    assert has_buy_sell(df)
    assert 'trade_log_BUY.csv' in caplog.text
    assert 'trade_log_SELL.csv' in caplog.text
