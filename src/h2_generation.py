"""Controlled H2 scenarios: alternative smoking-related risk mechanisms.

The PLCOm2012 baseline covariate distribution is kept unchanged. S1/S2 reuse
published *components* from Remen et al. (2018), not their fitted full model.
Published case-control odds-ratio coefficients must not be interpreted as
calibrated 6-year risk coefficients; the hybrid is a synthetic stress test.
S3 interactions and their strengths are original simulation assumptions.

No column computed here other than the 11 predictors belongs in an ML feature set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from src.data_generation import generate_baseline

SCENARIOS = ("S0", "S1", "S2", "S3")

# Tammemägi et al. (2013): published coefficients in the PLCOm2012 logit.
_B_DURATION = 0.0317321
_B_QUIT = -0.0308572
_B_INTENSITY = -1.822606

# Remen et al. (2018), Table 4, MEN, joint duration+intensity model (model 5).
REMEN_MALE_DURATION = 0.072
REMEN_MALE_LOG_INTENSITY = 0.64
# Remen et al. (2018), Table 4, MEN, CSI-only model (model 4).
REMEN_MALE_CSI = 1.31
CSI_HALF_LIFE = 26.0
CSI_LAG_MALE = 1.0

# Designed synthetic interaction strengths, NOT published epidemiological estimates.
INTERACTION_DURATION_INTENSITY = -0.18
INTERACTION_INTENSITY_QUIT = -0.15


def csi_index(d, c, q, *, half_life: float = CSI_HALF_LIFE,
              lag: float = CSI_LAG_MALE) -> np.ndarray:
    """Published CSI shape with average cigarettes/day and time since quitting.

    CSI = (1 - 0.5 ** (d_star / tau)) * 0.5 ** (q_star / tau) * log(c+1)
    q_star = max(q-lag, 0)
    d_star = max(d+q-lag, 0) - q_star

    Reference: Remen et al. (2018), Methods, based on Leffondré et al.
    """
    d = np.asarray(d, dtype=float)
    c = np.asarray(c, dtype=float)
    q = np.asarray(q, dtype=float)
    if half_life <= 0 or lag < 0:
        raise ValueError("half_life must be positive and lag nonnegative")
    if np.any(d < 0) or np.any(c <= 0) or np.any(q < 0):
        raise ValueError("Invalid smoking duration/intensity/cessation")
    q_star = np.maximum(q - lag, 0.0)
    d_star = np.maximum(d + q - lag, 0.0) - q_star
    return (1.0 - np.power(0.5, d_star / half_life)) * (
        np.power(0.5, q_star / half_life)
    ) * np.log1p(c)


def _smoking_terms_plco(d: np.ndarray, c: np.ndarray, q: np.ndarray):
    """Original duration, intensity, quit-time terms; current-smoking indicator stays."""
    duration = _B_DURATION * (d - 27.0)
    intensity = _B_INTENSITY * (np.power(c / 10.0, -1) - 0.4021541613)
    quit_time = _B_QUIT * (q - 10.0)
    return duration, intensity, quit_time


def _scenario_delta(df: pd.DataFrame, scenario: str) -> np.ndarray:
    d = df["smoking_years"].to_numpy(dtype=float)
    c = df["cigs_per_day"].to_numpy(dtype=float)
    q = df["quit_years"].to_numpy(dtype=float)
    s = df["is_current_smoker"].to_numpy(dtype=float)
    old_duration, old_intensity, old_quit = _smoking_terms_plco(d, c, q)

    if scenario == "S0":
        return np.zeros(len(df), dtype=float)
    if scenario == "S1":
        # Both coefficients belong to the SAME published adjusted joint model.
        # The new term replaces PLCO duration+intensity, keeping PLCO quit-time.
        new_term = REMEN_MALE_DURATION * d + REMEN_MALE_LOG_INTENSITY * np.log(c)
        return new_term - old_duration - old_intensity
    if scenario == "S2":
        # Remen male CSI model replaces duration+intensity+quit terms together.
        new_term = REMEN_MALE_CSI * csi_index(d, c, q)
        return new_term - old_duration - old_intensity - old_quit
    if scenario == "S3":
        # Explicit *synthetic* interactions, centered around reference values.
        i1 = ((d - 27.0) / 10.0) * np.log((c + 1.0) / 11.0)
        i2 = (1.0 - s) * np.log((c + 1.0) / 11.0) * ((q - 10.0) / 10.0)
        return (INTERACTION_DURATION_INTENSITY * i1
                + INTERACTION_INTENSITY_QUIT * i2)
    raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")


def apply_h2_scenario(
    baseline: pd.DataFrame, *, scenario: str, outcome_seed: int = 42,
    calibrate_prevalence: bool = True,
) -> pd.DataFrame:
    """Apply scenario to the SAME patient covariates, with common Bernoulli draws.

    The scalar intercept is solved to match mean PLCO risk on this synthetic
    *covariate pool*, before any train/test split. It is a simulation-control
    assumption, not a fitted clinical intercept. No held-out labels are used.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    needed = {"age", "smoking_years", "quit_years", "cigs_per_day",
              "is_current_smoker", "logit_base"}
    missing = needed.difference(baseline.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if baseline.empty:
        raise ValueError("Baseline data cannot be empty")

    df = baseline.copy(deep=True)
    plco_logit = df["logit_base"].to_numpy(dtype=float)
    delta = _scenario_delta(df, scenario)
    target = float(expit(plco_logit).mean())

    alpha = 0.0
    if scenario != "S0" and calibrate_prevalence:
        alpha = float(brentq(
            lambda a: float(expit(plco_logit + delta + a).mean() - target),
            -30.0, 30.0,
        ))

    df["delta_logit"] = delta
    df["calibration_shift"] = alpha
    df["logit_true"] = plco_logit + delta + alpha
    df["risk"] = expit(df["logit_true"].to_numpy())
    # Each scenario uses the SAME independent U(0,1) sequence for a given seed.
    # This pairs outcomes across scenarios without copying baseline's y column.
    uniform = np.random.default_rng(
        np.random.SeedSequence([int(outcome_seed), 2026])
    ).random(len(df))
    df["y"] = (uniform < df["risk"].to_numpy()).astype(int)
    df.attrs["h2_scenario"] = scenario
    df.attrs["h2_calibration_shift"] = alpha
    df.attrs["h2_target_prevalence"] = target
    return df


def generate_h2_scenario(n: int = 50_000, seed: int = 42,
                         scenario: str = "S0") -> pd.DataFrame:
    """Generate baseline X and scenario-specific outcome with fixed seeds."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    return apply_h2_scenario(
        generate_baseline(n=n, seed=seed),
        scenario=scenario,
        outcome_seed=seed,
    )
