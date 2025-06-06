import os
import sys
import pandas as pd
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from strategy import trend_filter


@pytest.mark.parametrize(
    "trend,expected_long,expected_short",
    [
        ("UP", 1, 0),
        ("DOWN", 0, 1),
        ("NEUTRAL", 0, 0),
    ],
)
def test_apply_trend_filter(trend, expected_long, expected_short):
    df = pd.DataFrame(
        {
            "Trend_Zone": [trend],
            "Entry_Long": [1],
            "Entry_Short": [1],
        }
    )
    result = trend_filter.apply_trend_filter(df)
    assert result["Entry_Long"].iloc[0] == expected_long
    assert result["Entry_Short"].iloc[0] == expected_short


def test_apply_trend_filter_no_trend_zone():
    df = pd.DataFrame({"Entry_Long": [1], "Entry_Short": [1]})
    result = trend_filter.apply_trend_filter(df)
    # Should return a copy unchanged when Trend_Zone column missing
    assert result.equals(df)
    assert result is not df


@pytest.mark.parametrize("missing", ["Entry_Long", "Entry_Short"])
def test_apply_trend_filter_missing_required_columns(missing):
    df = pd.DataFrame({"Trend_Zone": ["UP"], "Entry_Long": [1], "Entry_Short": [1]})
    df.drop(columns=[missing], inplace=True)
    with pytest.raises(KeyError):
        trend_filter.apply_trend_filter(df)


def test_apply_trend_filter_invalid_type():
    with pytest.raises(TypeError):
        trend_filter.apply_trend_filter([1, 2, 3])
