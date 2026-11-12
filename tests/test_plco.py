
"""Tests for the published PLCOm2012 implementation."""

import pytest

from src.plco import plco_logit, plco_risk


@pytest.fixture
def patient():
    """Reference former smoker."""
    return {
        "age": 62,
        "bmi": 27,
        "education": 4,
        "smoking_years": 27,
        "quit_years": 10,
        "cigs_per_day": 20,
        "is_current_smoker": 0,
        "copd": 0,
        "personal_cancer": 0,
        "family_lung_cancer": 0,
        "race": "White",
    }


def test_former_smoker_reference(patient):
    """Check logit and risk for a reference patient."""
    assert plco_logit(patient) == pytest.approx(
        -4.710840412689652
    )
    assert plco_risk(patient) == pytest.approx(
        0.008916985018390567
    )


def test_current_smoker_includes_centered_quit_term(patient):
    """Check the quit-years term for current smokers."""
    current = {
        **patient,
        "is_current_smoker": 1,
        "quit_years": 0,
    }

    assert plco_logit(current) == pytest.approx(
        -4.142525312689653
    )
    assert plco_risk(current) == pytest.approx(
        0.015634376073833757
    )


def test_reference_example_from_r_package(patient):
    """Compare with an independently published example."""
    sample = {**patient, "cigs_per_day": 80}

    assert plco_risk(sample) == pytest.approx(
        0.01750922,
        abs=1e-8,
    )


def test_copd_coefficient(patient):
    """COPD must add exactly its published coefficient."""
    with_copd = {**patient, "copd": 1}

    difference = (
        plco_logit(with_copd) - plco_logit(patient)
    )

    assert difference == pytest.approx(0.3553063)


def test_reject_nonpositive_cigarettes(patient):
    with pytest.raises(ValueError):
        plco_risk({**patient, "cigs_per_day": 0})


def test_reject_inconsistent_quit_years(patient):
    with pytest.raises(ValueError):
        plco_risk({
            **patient,
            "is_current_smoker": 1,
            "quit_years": 10,
        })


def test_reject_unknown_race(patient):
    with pytest.raises(ValueError):
        plco_risk({**patient, "race": "Unknown"})
