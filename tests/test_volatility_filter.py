import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.strategy import passes_volatility_filter


def test_volatility_filter_pass():
    assert passes_volatility_filter(1.2)


def test_volatility_filter_block():
    assert not passes_volatility_filter(0.8)


def test_volatility_filter_nan():
    assert not passes_volatility_filter(np.nan)
