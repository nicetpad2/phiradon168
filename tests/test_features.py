import pandas as pd
import pytest
from src.features import add_simple_features


def test_add_simple_features():
    df = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
    result = add_simple_features(df)
    assert 'sma_5' in result.columns
    assert result['sma_5'].iloc[4] == pytest.approx(3.0)
