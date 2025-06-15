import pandas as pd
from strategy import strategy as strat
import strategy.order_management as order_management


def test_run_backtest_short_trade(monkeypatch):
    df = pd.DataFrame({
        "Close": [1.0, 0.9, 0.8, 0.7],
        "MACD_hist": [-0.1] * 4,
        "RSI": [40] * 4,
        "MA_fast": [1.0, 0.95, 0.9, 0.85],
        "MA_slow": [1.1, 1.1, 1.1, 1.1],
        "ATR_14": [0.1] * 4,
    })
    monkeypatch.setattr(order_management.OrderManager, "place_order", lambda self, o, t: order_management.OrderStatus.OPEN)
    trades = strat.run_backtest(df, 1000.0, allow_short=True)
    assert len(trades) == 1
    assert trades[0]["profit"] > 0
