import numpy as np
from src.features import is_volume_spike


def test_is_volume_spike_true():
    assert is_volume_spike(150, 90)


def test_is_volume_spike_false():
    assert not is_volume_spike(100, 100)


def test_is_volume_spike_invalid():
    assert not is_volume_spike(np.nan, 50)
