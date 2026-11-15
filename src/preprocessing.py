"""
Feature selection, stratified splitting and preprocessing for ML experiments.

Responsibilities
----------------
- Select only observable patient predictors; exclude generator-only values.
- Create reproducible, disjoint train/validation/test splits.
- Define transformations to fit on training data inside an ML pipeline.

Notes
-----
- The validation set is used to choose a classification threshold.
- The test set is reserved for the final evaluation.
- Fit the preprocessor ONLY on X_train, never on validation or test data.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = (
    "age",
    "bmi",
    "education",       # Ordinal level treated numerically in the first baseline.
    "cigs_per_day",
    "smoking_years",
    "quit_years",
)
BINARY_FEATURES = (
    "copd",
    "personal_cancer",
    "family_lung_cancer",
    "is_current_smoker",
)
CATEGORICAL_FEATURES = ("race",)
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DatasetSplits:
    """Feature matrices and binary labels for all three disjoint sets."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def get_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select ML predictors X and labels y without generator-derived leakage."""
    required = set(FEATURE_COLUMNS) | {"y"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not df.index.is_unique:
        raise ValueError("DataFrame index must be unique")
    if df.loc[:, list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Missing predictor values are not supported yet")
    if not df["y"].isin((0, 1)).all():
        raise ValueError("y must consist only of 0 and 1")

    # The allowlist intentionally excludes risk, logit_base and logit_true.
    X = df.loc[:, list(FEATURE_COLUMNS)].copy()
    y = df["y"].astype(int).copy()
    return X, y


def split_dataset(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.20,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> DatasetSplits:
    """Create a stratified train/validation/test split, by default 60/20/20."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("val_fraction and test_fraction must be in (0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    X, y = get_features_target(df)
    if y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError("Need at least 5 observations of each class for splitting")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_fraction + test_fraction,
        stratify=y,
        random_state=seed,
    )
    # Among the temporary holdout samples, separate validation from test.
    relative_test_fraction = test_fraction / (val_fraction + test_fraction)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_fraction,
        stratify=y_temp,
        random_state=seed + 1,
    )

    return DatasetSplits(X_train, X_val, X_test, y_train, y_val, y_test)


def make_preprocessor(*, scale_numeric: bool = True) -> ColumnTransformer:
    """Build an unfitted preprocessor; fit it on training data only."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler() if scale_numeric else "passthrough",
                list(NUMERIC_FEATURES),
            ),
            ("binary", "passthrough", list(BINARY_FEATURES)),
            (
                "race",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
