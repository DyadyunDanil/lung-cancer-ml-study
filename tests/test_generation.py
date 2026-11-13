
"""
Tests for synthetic lung-cancer data generators.

Generators
----------
1. generate_article_legacy(n, seed)
   - Checks reproducibility of the recovered generator.
   - Preserves historical results, including known inconsistencies.

2. generate_baseline(n, seed)
   - Checks reproducibility and valid smoking histories.
   - Verifies that risks follow the published PLCOm2012 formula.
   - Ensures no additional risk interactions are introduced.

Validation
----------
- Dataset size and deterministic generation.
- Valid probabilities and binary outcomes.
- Consistency of smoking status, duration, and quit years.
- Correct calculation of baseline risk.

Notes
-----
- Legacy regression tests preserve historical behavior.
- Baseline tests enforce the corrected data-generation rules.
"""

import numpy as np
import pandas as pd
import pytest

from scipy.special import expit

from src.data_generation import (
    generate_article_legacy,
    generate_baseline,
)
from src.plco import plco_logit


# ---------------------------------------------------------
# 1. Legacy generator tests
# ---------------------------------------------------------

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
    """Preserve the recovered generator's reference results."""
    df = generate_article_legacy(n=10_000, seed=42)

    inconsistent_current = (
        (df["is_current_smoker"] == 1)
        & (df["quit_years"] > 0)
    )

    assert int(df["y"].sum()) == 121
    assert int(inconsistent_current.sum()) == 1607


# ---------------------------------------------------------
# 2. Baseline generator tests
# ---------------------------------------------------------

def test_baseline_reproducibility():
    """The same seed must produce identical datasets."""
    first = generate_baseline(n=300, seed=42)
    second = generate_baseline(n=300, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_baseline_smoking_history():
    """Smoking histories must satisfy logical constraints."""
    df = generate_baseline(n=10_000, seed=42)

    current = df["is_current_smoker"] == 1

    # Current smokers have not quit.
    assert (df.loc[current, "quit_years"] == 0).all()

    # Former smokers have positive quit time.
    assert (df.loc[~current, "quit_years"] >= 1).all()

    # Smoking cannot start before age 15 (simulation assumption).
    starting_age = (
        df["age"]
        - df["smoking_years"]
        - df["quit_years"]
    )

    assert (starting_age >= 15).all()
    assert (df["smoking_years"] >= 1).all()
    assert (df["cigs_per_day"] >= 1).all()


def test_baseline_uses_published_plco():
    """Risk must follow PLCOm2012 without extra interactions."""
    df = generate_baseline(n=200, seed=42)

    expected_logit = df.apply(plco_logit, axis=1)

    np.testing.assert_allclose(
        df["logit_base"],
        expected_logit,
    )

    # No artificial interactions in the baseline scenario.
    np.testing.assert_allclose(
        df["logit_true"],
        df["logit_base"],
    )

    np.testing.assert_allclose(
        df["risk"],
        expit(expected_logit.to_numpy()),
    )

    assert df["risk"].between(0, 1).all()
    assert df["y"].isin([0, 1]).all()


def test_baseline_rejects_invalid_size():
    """The generator must reject nonpositive sample sizes."""
    with pytest.raises(ValueError):
        generate_baseline(n=0, seed=42)

    with pytest.raises(ValueError):
        generate_baseline(n=-100, seed=42)
