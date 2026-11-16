"""H2 scientific mechanisms, reproducibility and leakage checks."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.data_generation import generate_baseline
from src.h2_generation import (
    REMEN_MALE_CSI, REMEN_MALE_DURATION, REMEN_MALE_LOG_INTENSITY,
    SCENARIOS, apply_h2_scenario, csi_index, generate_h2_scenario,
)
from src.preprocessing import FEATURE_COLUMNS, get_features_target


@pytest.fixture(scope="module")
def patients():
    return generate_baseline(n=2000, seed=73)


def test_csi_formula_manual_values():
    # Male reference tau=26, lag=1; current smoker, d=27, c=10, q=0.
    result = float(csi_index([27], [10], [0])[0])
    expected = (1 - 0.5 ** (26 / 26)) * np.log(11)
    assert result == pytest.approx(expected)


def test_s0_reproduces_plco_probabilities(patients):
    df = apply_h2_scenario(patients, scenario="S0", outcome_seed=100)
    np.testing.assert_allclose(df.risk, patients.risk, rtol=0, atol=1e-14)
    np.testing.assert_allclose(df.logit_true, patients.logit_base, rtol=0, atol=1e-14)


def test_remen_joint_coefficients_from_same_model(patients):
    df = apply_h2_scenario(patients, scenario="S1", outcome_seed=100)
    d = patients.smoking_years.to_numpy()
    c = patients.cigs_per_day.to_numpy()
    q = patients.quit_years.to_numpy()
    old_d = 0.0317321 * (d - 27)
    old_c = -1.822606 * (10.0 / c - 0.4021541613)
    expected = REMEN_MALE_DURATION*d + REMEN_MALE_LOG_INTENSITY*np.log(c) - old_d - old_c
    np.testing.assert_allclose(df.delta_logit, expected)


def test_remen_csi_coefficient(patients):
    df = apply_h2_scenario(patients, scenario="S2", outcome_seed=100)
    d = patients.smoking_years.to_numpy()
    c = patients.cigs_per_day.to_numpy()
    q = patients.quit_years.to_numpy()
    old = 0.0317321*(d-27) - 1.822606*(10.0/c-0.4021541613) - 0.0308572*(q-10)
    np.testing.assert_allclose(df.delta_logit, REMEN_MALE_CSI * csi_index(d, c, q) - old)


def test_prevalence_mean_matched_and_reproducible(patients):
    reference = float(patients.risk.mean())
    for scenario in SCENARIOS:
        a = apply_h2_scenario(patients, scenario=scenario, outcome_seed=99)
        b = apply_h2_scenario(patients, scenario=scenario, outcome_seed=99)
        assert np.isclose(a.risk.mean(), reference, atol=1e-12)
        pd.testing.assert_frame_equal(a, b)
        assert (a.risk.between(0, 1, inclusive="neither")).all()
        assert (a.y.isin([0,1])).all()


def test_same_patient_x_and_only_allowed_features(patients):
    for scenario in SCENARIOS:
        df = apply_h2_scenario(patients, scenario=scenario, outcome_seed=99)
        pd.testing.assert_frame_equal(df[list(FEATURE_COLUMNS)], patients[list(FEATURE_COLUMNS)])
        X, y = get_features_target(df)
        assert list(X.columns) == list(FEATURE_COLUMNS)
        assert not ({"risk", "logit_true", "logit_base", "delta_logit", "calibration_shift"} & set(X))


def test_invalid_scenario_and_n():
    with pytest.raises(ValueError, match="Unknown scenario"):
        generate_h2_scenario(100, seed=1, scenario="invalid")
    with pytest.raises(ValueError):
        generate_h2_scenario(0, seed=1, scenario="S0")


def test_seed_is_same_across_calls():
    a = generate_h2_scenario(150, 7, "S3")
    b = generate_h2_scenario(150, 7, "S3")
    pd.testing.assert_frame_equal(a, b)
