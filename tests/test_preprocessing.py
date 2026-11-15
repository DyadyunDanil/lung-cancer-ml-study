"""
Tests for dataset splitting and feature preprocessing.

Checks
------
- Strict feature allowlist excludes true risk and generator logits.
- Disjoint, stratified and reproducible train/validation/test splits.
- Preprocessing is fitted only on training observations.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURE_COLUMNS,
    get_features_target,
    make_preprocessor,
    split_dataset,
)


@pytest.fixture
def example_df():
    rng = np.random.default_rng(7)
    n = 200
    return pd.DataFrame({
        "age": rng.integers(55, 75, n),
        "bmi": rng.normal(27, 4, n),
        "education": rng.integers(1, 7, n),
        "copd": rng.integers(0, 2, n),
        "personal_cancer": rng.integers(0, 2, n),
        "family_lung_cancer": rng.integers(0, 2, n),
        "is_current_smoker": rng.integers(0, 2, n),
        "cigs_per_day": rng.integers(1, 30, n),
        "smoking_years": rng.integers(1, 40, n),
        "quit_years": rng.integers(0, 15, n),
        "race": rng.choice(["White", "Black", "Asian"], n),
        "logit_base": np.full(n, 100.0),
        "logit_true": np.full(n, 100.0),
        "risk": np.full(n, 0.99),
        "y": np.array([0] * 180 + [1] * 20),
    })


def test_feature_allowlist_excludes_generator_outputs(example_df):
    X, y = get_features_target(example_df)
    assert X.columns.tolist() == list(FEATURE_COLUMNS)
    assert not {"risk", "logit_base", "logit_true", "y"} & set(X.columns)
    assert len(y) == len(example_df)


def test_split_is_stratified_disjoint_and_repeatable(example_df):
    a = split_dataset(example_df, seed=42)
    b = split_dataset(example_df, seed=42)

    assert (len(a.X_train), len(a.X_val), len(a.X_test)) == (120, 40, 40)
    train_ids, val_ids, test_ids = map(
        lambda x: set(x.index), (a.X_train, a.X_val, a.X_test)
    )
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
    assert len(train_ids | val_ids | test_ids) == len(example_df)
    for X, y in [
        (a.X_train, a.y_train),
        (a.X_val, a.y_val),
        (a.X_test, a.y_test),
    ]:
        assert X.index.equals(y.index)
        assert y.mean() == pytest.approx(0.10)

    pd.testing.assert_frame_equal(a.X_train, b.X_train)
    pd.testing.assert_series_equal(a.y_test, b.y_test)


def test_preprocessor_only_learns_from_train(example_df):
    X, _ = get_features_target(example_df)
    train = X.iloc[:10].copy()
    train["age"] = np.arange(55, 65)
    val = X.iloc[10:12].copy()
    val["age"] = [150, 160]

    preprocessor = make_preprocessor()
    train_transformed = preprocessor.fit_transform(train)
    val_transformed = preprocessor.transform(val)

    # First feature is standardized age; unseen high ages stay far from 0.
    assert train_transformed[:, 0].mean() == pytest.approx(0.0, abs=1e-12)
    assert (val_transformed[:, 0] > 20).all()
    assert train_transformed.shape[1] == val_transformed.shape[1]


def test_invalid_split_and_target_rejected(example_df):
    with pytest.raises(ValueError):
        split_dataset(example_df, val_fraction=0.7, test_fraction=0.3)
    bad = example_df.copy()
    bad.loc[0, "y"] = 2
    with pytest.raises(ValueError):
        get_features_target(bad)
