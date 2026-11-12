
"""Tests for the reconstructed article generator."""

import pandas as pd

from src.data_generation import generate_article_legacy


def test_number_of_rows():
    df = generate_article_legacy(n=100, seed=42)
    assert len(df) == 100


def test_reproducibility():
    first = generate_article_legacy(n=100, seed=42)
    second = generate_article_legacy(n=100, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_probability_range():
    df = generate_article_legacy(n=100, seed=42)

    assert df["risk"].between(0, 1).all()
    assert df["y"].isin([0, 1]).all()


def test_article_seed_42_regression():
    """Lock in results of the recovered historical code."""
    df = generate_article_legacy(n=10_000, seed=42)

    inconsistent_current = (
        (df["is_current_smoker"] == 1)
        & (df["quit_years"] > 0)
    )

    assert int(df["y"].sum()) == 121
    assert int(inconsistent_current.sum()) == 1607
