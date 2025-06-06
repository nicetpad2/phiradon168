import os
import pandas as pd
import pytest
import src.training as training


def test_real_train_func_empty_trade_log(tmp_path, monkeypatch):
    trade_path = tmp_path / 'trade.csv'
    m1_path = tmp_path / 'm1.csv'
    # create empty trade log
    pd.DataFrame(columns=['profit']).to_csv(trade_path, index=False)
    pd.DataFrame({
        'Open': [1, 2, 3],
        'High': [2, 3, 4],
        'Low': [0, 1, 2],
        'Close': [1.5, 2.5, 3.5]
    }).to_csv(m1_path, index=False)

    monkeypatch.setattr(training, 'CatBoostClassifier', None, raising=False)

    with pytest.raises(ValueError):
        training.real_train_func(
            output_dir=str(tmp_path),
            trade_log_path=str(trade_path),
            m1_path=str(m1_path),
            seed=42,
        )
