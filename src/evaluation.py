"""
Validation-threshold selection and evaluation of binary risk predictions.

Protocol
--------
1. Fit a model on training data.
2. Choose its decision threshold using validation labels and scores.
3. Evaluate on held-out test data with that SAME threshold.

Metrics
-------
- Recall and specificity at the validation-selected threshold.
- ROC AUC, average precision (AP), Brier score.
- Standardized partial ROC AUC for false-positive rate <= 0.10.

Notes
-----
- Test specificity is reported, not constrained or retuned.
- AP is not identical to trapezoidal area under the PR curve.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def _check_binary_data(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary labels and finite risk probabilities."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and probabilities must be nonempty 1-D arrays of equal length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if np.unique(y).size < 2:
        raise ValueError("Both classes are required for these metrics")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite values between 0 and 1")
    return y.astype(int), p


def select_threshold_at_specificity(
    y_val, probabilities_val, *, min_specificity: float = 0.90
) -> float:
    """Choose the LOWEST validation threshold giving specificity >= target.

    Prediction convention: positive if probability >= threshold.
    Lowest feasible threshold maximizes validation recall subject to the
    specificity constraint. Tied scores are handled as a group by roc_curve.
    """
    if not 0 < min_specificity <= 1:
        raise ValueError("min_specificity must lie in (0, 1]")
    y, p = _check_binary_data(y_val, probabilities_val)
    fpr, _tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    feasible = np.flatnonzero(fpr <= 1 - min_specificity + 1e-12)
    chosen = float(thresholds[feasible[-1]])

    if np.isinf(chosen):
        # ROC's sentinel predicts nobody; keep an equivalent finite threshold.
        return float(np.nextafter(p.max(), np.inf))
    return chosen


def evaluate_binary(y_true, probabilities, *, threshold: float) -> dict:
    """Evaluate probabilities and fixed-threshold predictions on ONE dataset."""
    y, p = _check_binary_data(y_true, probabilities)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "pauc_standardized_fpr_0_10": float(roc_auc_score(y, p, max_fpr=0.10)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
    }
