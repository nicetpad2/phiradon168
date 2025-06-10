import os
import pandas as pd
import src.training as training


def test_real_train_func_with_real_data(tmp_path, monkeypatch):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    profits = [1, -1] * 5
    pd.DataFrame({'profit': profits}).to_csv(trade_path, index=False)
    pd.DataFrame({
        'Open': range(1, 11),
        'High': range(2, 12),
        'Low': range(0, 10),
        'Close': [i + 0.5 for i in range(1, 11)]
    }).to_csv(m1_path, index=False)

    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)

    out_dir = tmp_path / 'out'
    res = training.real_train_func(
        output_dir=str(out_dir),
        trade_log_path=str(trade_path),
        m1_path=str(m1_path),
        seed=123
    )
    assert 'model_path' in res
    assert os.path.exists(res['model_path']['model'])
    assert set(res['features']) == {'Open', 'High', 'Low', 'Close'}
