"""Utilities to help prevent data leakage when splitting data."""

from __future__ import annotations

import hashlib
import pandas as pd

__all__ = ["hash_df", "timestamp_split", "assert_no_overlap"]


def hash_df(df: pd.DataFrame) -> str:
    """Return a hash representing the DataFrame contents."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def timestamp_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets using ``loc`` to avoid leakage."""
    df_sorted = df.sort_index()
    train = df_sorted.loc[train_start:train_end].copy()
    test = df_sorted.loc[test_start:test_end].copy()
    return train, test


def assert_no_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if train/test indices overlap."""
    overlap = train_df.index.intersection(test_df.index)
    if not overlap.empty:
        raise ValueError(f"Data leakage detected: {len(overlap)} overlapping timestamps")
