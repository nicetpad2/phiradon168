import numpy as np
import sys

from src.training import train_lstm_sequence


def test_train_lstm_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, 'tensorflow', None)
    X = np.random.rand(4, 3, 2).astype('float32')
    y = np.array([0, 1, 0, 1])
    model = train_lstm_sequence(X, y, epochs=1)
    assert hasattr(model, 'predict')

