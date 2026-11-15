"""
Tests for validation-only threshold selection and binary metrics.

Checks
------
- Validation specificity constraint and maximal feasible recall.
- Correct treatment of tied risk scores and constant predictions.
- Correct confusion-matrix metrics at an unchanged threshold.
- Rejection of invalid class labels or risk scores.
"""

import numpy as np
import pytest

from src.evaluation import evaluate_binary, select_threshold_at_specificity


def test_threshold_hits_validation_specificity_target():
    y = np.array([0] * 10 + [1] * 2)
    p = np.array([0.1] * 8 + [0.4, 0.6] + [0.5, 0.7])

    threshold = select_threshold_at_specificity(y, p, min_specificity=0.90)
    assert threshold == pytest.approx(0.5)
    report = evaluate_binary(y, p, threshold=threshold)
    assert report["specificity"] == pytest.approx(0.90)
    assert report["recall"] == pytest.approx(1.0)


def test_threshold_handles_tied_negative_scores():
    y = np.array([0] * 10 + [1] * 2)
    p = np.array([0.1] * 8 + [0.6, 0.6] + [0.5, 0.7])
    threshold = select_threshold_at_specificity(y, p, min_specificity=0.90)
    assert threshold == pytest.approx(0.7)
    assert evaluate_binary(y, p, threshold=threshold)["specificity"] == 1.0


def test_threshold_can_predict_nobody():
    y = np.array([0, 0, 0, 1, 1])
    p = np.full(len(y), 0.5)
    threshold = select_threshold_at_specificity(y, p, min_specificity=0.90)
    assert threshold > 0.5
    report = evaluate_binary(y, p, threshold=threshold)
    assert report["specificity"] == 1.0
    assert report["recall"] == 0.0


def test_confusion_metrics_and_probabilistic_metrics():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.4, 0.9, 0.3])
    report = evaluate_binary(y, p, threshold=0.5)
    assert (report["tn"], report["fp"], report["fn"], report["tp"]) == (3, 1, 1, 1)
    assert report["recall"] == pytest.approx(0.5)
    assert report["specificity"] == pytest.approx(0.75)
    assert report["precision"] == pytest.approx(0.5)
    assert 0 <= report["roc_auc"] <= 1
    assert 0 <= report["average_precision"] <= 1
    assert 0 <= report["brier_score"] <= 1
    assert 0.5 <= report["pauc_standardized_fpr_0_10"] <= 1


def test_invalid_data_rejected():
    with pytest.raises(ValueError):
        select_threshold_at_specificity([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError):
        select_threshold_at_specificity([0, 1], [0.2, float("nan")])
    with pytest.raises(ValueError):
        evaluate_binary([0, 1], [0.1, 1.2], threshold=0.5)
    with pytest.raises(ValueError):
        select_threshold_at_specificity([0, 1], [0.1, 0.9], min_specificity=1.1)
