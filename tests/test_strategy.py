import pandas as pd
from src.strategy import simple_strategy


def test_simple_strategy():
    df = pd.DataFrame({'Close': [1, 2, 3, 2, 1, 2, 3]})
    buys, sells = simple_strategy(df)
    assert isinstance(buys, int)
    assert isinstance(sells, int)
