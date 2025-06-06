import os
import sys
import pandas as pd
import logging
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.utils.trade_splitter import split_trade_log, has_buy_sell, normalize_side


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


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("BUY", "BUY"),
        ("B", "BUY"),
        ("1", "BUY"),
        ("sell", "SELL"),
        ("S", "SELL"),
        ("0", "SELL"),
        ("-1", "SELL"),
        ("HOLD", "NORMAL"),
        (None, "NORMAL"),
    ],
)
def test_normalize_side(inp, expected):
    assert normalize_side(inp) == expected


def test_has_buy_sell_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    assert has_buy_sell(df) is False


def test_split_trade_log_empty_warning(tmp_path, caplog):
    out_dir = tmp_path / "out"
    with caplog.at_level(logging.WARNING):
        split_trade_log(pd.DataFrame(), str(out_dir))
    assert "[QA-WARNING]" in caplog.text
    assert not any(out_dir.glob("*.csv"))
