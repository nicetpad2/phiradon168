import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.utils.model_utils import save_model, load_model, evaluate_model, predict


def test_save_load_and_predict(tmp_path):
    X = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
    y = [0, 1]
    model = LogisticRegression()
    model.fit(X, y)
    path = tmp_path / 'm.pkl'
    save_model(model, str(path))
    loaded = load_model(str(path))
    prob = predict(loaded, X.iloc[[0]])
    assert 0.0 <= prob <= 1.0
    acc, auc = evaluate_model(loaded, X, y)
    assert acc >= 0
