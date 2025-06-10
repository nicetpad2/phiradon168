import pytest

from src.utils.auto_train_meta_classifiers import auto_train_meta_classifiers


def test_auto_train_meta_classifiers_returns_none():
    """Ensure stub returns None without error."""
    result = auto_train_meta_classifiers({}, [])
    assert result is None
