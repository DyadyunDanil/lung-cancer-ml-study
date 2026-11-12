
"""
Published PLCOm2012 six-year lung-cancer risk model.

Source:
Tammemägi et al. (2013)
DOI: 10.1056/NEJMoa1211776

For current smokers, quit_years = 0.
The centered quit-years term is included for all patients.

This implementation is separate from the reconstructed
generator used in the earlier research article.
"""

from collections.abc import Mapping

from scipy.special import expit


RACE_COEFFICIENTS = {
    "White": 0.0,
    "Black": 0.3944778,
    "Hispanic": -0.7434744,
    "Asian": -0.466585,
    "American Indian or Alaskan Native": 0.0,
    "Native Hawaiian or Pacific Islander": 1.027152,
}


def plco_logit(row: Mapping) -> float:
    """Calculate the PLCOm2012 logit for one patient."""

    cigs = float(row["cigs_per_day"])
    quit_years = float(row["quit_years"])
    current = row["is_current_smoker"]
    race = row["race"]

    # Validate important input values.
    if cigs <= 0:
        raise ValueError("cigs_per_day must be positive")

    if current not in (0, 1):
        raise ValueError("is_current_smoker must be 0 or 1")

    if quit_years < 0:
        raise ValueError("quit_years cannot be negative")

    if current == 1 and quit_years != 0:
        raise ValueError(
            "Current smokers must have quit_years == 0"
        )

    if race not in RACE_COEFFICIENTS:
        raise ValueError(f"Unknown race category: {race}")

    # Intercept and centered continuous features.
    logit = -4.532506

    logit += 0.0778868 * (float(row["age"]) - 62)
    logit -= 0.0274194 * (float(row["bmi"]) - 27)
    logit -= 0.0812744 * (float(row["education"]) - 4)
    logit += 0.0317321 * (float(row["smoking_years"]) - 27)

    # Published formula: included for all smokers.
    # Current smokers have quit_years = 0.
    logit -= 0.0308572 * (quit_years - 10)

    # Nonlinear smoking-intensity transformation.
    logit -= 1.822606 * ((cigs / 10) ** (-1) - 0.4021541613)

    # Binary risk factors.
    logit += 0.3553063 * float(row["copd"])
    logit += 0.4589971 * float(row["personal_cancer"])
    logit += 0.587185 * float(row["family_lung_cancer"])
    logit += 0.2597431 * current

    # Race/ethnicity coefficient.
    logit += RACE_COEFFICIENTS[race]

    return float(logit)


def plco_risk(row: Mapping) -> float:
    """Calculate six-year lung-cancer risk probability."""
    return float(expit(plco_logit(row)))
