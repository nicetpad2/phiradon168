import os
import src.training as training
import pandas as pd


def test_real_train_func_accepts_seed(tmp_path):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    pd.DataFrame({'profit': [1, -1, 2]}).to_csv(trade_path, index=False)
    pd.DataFrame({
        'Open': [1, 2, 3],
        'High': [2, 3, 4],
        'Low': [0, 1, 2],
        'Close': [1.5, 2.5, 3.5]
    }).to_csv(m1_path, index=False)

    res = training.real_train_func(
        output_dir=str(tmp_path),
        trade_log_path=str(trade_path),
        m1_path=str(m1_path),
        seed=123,
    )
    assert 'model_path' in res
    assert os.path.exists(res['model_path']['model'])
    assert 'metrics' in res
