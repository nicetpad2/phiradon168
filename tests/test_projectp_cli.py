import ProjectP as proj
import pytest

def test_parse_args_modes():
    assert proj.parse_args(["--mode", "sweep"]).mode == "sweep"
    assert proj.parse_args([]).mode == "preprocess"


def test_run_mode_invalid():
    with pytest.raises(ValueError):
        proj.run_mode("unknown")
