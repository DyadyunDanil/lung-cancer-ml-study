
"""
Synthetic data generation for lung-cancer risk experiments.

The legacy generator reconstructs the implementation found
after the publication of the earlier research article.

It intentionally preserves known inconsistencies.
Do not use it as the corrected baseline generator.
"""

import numpy as np
import pandas as pd


def _article_legacy_logit(row):
    """Reconstructed PLCO-like formula from the earlier code."""
    logit = -4.532506
    logit += 0.0778868 * (row["age"] - 62)
    logit -= 0.0274194 * (row["bmi"] - 27)
    logit -= 0.0812744 * (row["education"] - 4)
    logit += 0.0317321 * (row["smoking_years"] - 27)

    # Historical behavior: applied only to former smokers.
    if row["is_current_smoker"] == 0:
        logit -= 0.0308572 * (row["quit_years"] - 10)

    cigs = row["cigs_per_day"]
    if cigs > 0:
        transform = (cigs / 10) ** (-1) - 0.4021541613
        logit -= 1.822606 * transform

    logit += 0.3553063 * row["copd"]
    logit += 0.4589971 * row["personal_cancer"]
    logit += 0.587185 * row["family_lung_cancer"]
    logit += 0.2597431 * row["is_current_smoker"]

    race = row["race"]
    if race == "Black":
        logit += 0.3944778
    elif race == "Asian":
        logit -= 0.466585
    elif race == "Hispanic":
        logit -= 0.7434744
    elif race == "Native Hawaiian or Pacific Islander":
        logit += 1.027152

    return logit


def generate_article_legacy(n: int, seed: int) -> pd.DataFrame:
    """
    Reproduce the recovered synthetic-data generator.

    Known limitations:
    - quit_years is generated independently of smoking status;
    - smoking history may be inconsistent with age;
    - the recovered quit-years formula differs from the
      published PLCOm2012 implementation;
    - an artificial +1.2 logit interaction is added.

    Returns X features, intermediate probabilities and y
    in one DataFrame for historical reproducibility.
    """
    rng = np.random.RandomState(seed)

    data = {
        "age": rng.randint(55, 75, n),
        "bmi": np.clip(rng.normal(27, 5, n), 15, 50),
        "education": rng.choice(
            [1, 2, 3, 4, 5, 6],
            n, p=[0.05, 0.20, 0.25, 0.30, 0.15, 0.05]
        ),
        "copd": rng.choice([0, 1], n, p=[0.85, 0.15]),
        "personal_cancer": rng.choice(
            [0, 1], n, p=[0.92, 0.08]
        ),
        "family_lung_cancer": rng.choice(
            [0, 1], n, p=[0.88, 0.12]
        ),
        "is_current_smoker": rng.choice(
            [0, 1], n, p=[0.60, 0.40]
        ),
        "cigs_per_day": np.clip(
            rng.exponential(10, n), 1, 60
        ).astype(int),
        "smoking_years": np.clip(
            rng.normal(30, 10, n), 1, 60
        ).astype(int),
        # Preserved intentionally: independent of smoking status.
        "quit_years": np.where(
            rng.rand(n) < 0.4,
            np.clip(rng.normal(10, 5, n), 0, 40).astype(int),
            0
        ),
        "race": rng.choice(
            [
                "White",
                "Black",
                "Hispanic",
                "Asian",
                "Native Hawaiian or Pacific Islander",
            ],
            n,
            p=[0.70, 0.12, 0.09, 0.07, 0.02],
        ),
    }

    df = pd.DataFrame(data)

    df["logit_base"] = df.apply(
        _article_legacy_logit, axis=1
    )

    # Artificial interaction from the recovered article code.
    extra_risk = np.where(
        (df["age"] > 65) & (df["is_current_smoker"] == 1),
        1.2,
        0.0,
    )

    df["logit_true"] = df["logit_base"] + extra_risk

    # Preserve the original probability computation.
    df["risk"] = 1 / (1 + np.exp(-df["logit_true"]))
    df["y"] = rng.binomial(1, df["risk"])

    return df
