# Consolidated speed and accuracy tests
import pytest
from some_module import compute_metrics


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1, 2, 3], 2.0),
        ([10, 20], 15.0),
    ],
)
def test_compute_metrics_mean(values, expected):
    """Test compute_metrics returns correct mean"""
    result = compute_metrics(values)
    assert result['mean'] == pytest.approx(expected, rel=1e-6)
