"""Unit tests for metrics on hand-computed examples per spec Section 13."""

import numpy as np
import pytest
from aquasign.metrics import (
    compute_classification_metrics,
    compute_domain_gap,
    compute_risk_coverage_curve,
    selective_accuracy,
)


def test_metrics():
    # 6-sample hand-crafted example
    # Sample index:   0    1    2    3    4    5
    # y_true:         0    1    0    1    0    1
    # y_pred:         0    1    0    0    1    1
    # correct:        T    T    T    F    F    T
    # conf:          0.9  0.8  0.7  0.6  0.5  0.4
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    # 1. Overall accuracy: 4 / 6 = 2 / 3
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=2)
    assert np.isclose(metrics["accuracy"], 4.0 / 6.0)

    # 2. Selective accuracy at coverage 0.5:
    # Top round(6 * 0.5) = 3 samples: indices [0, 1, 2] -> all correct -> 3/3 = 1.0
    sel_acc_50 = selective_accuracy(y_true, y_pred, conf, coverage=0.5)
    assert np.isclose(sel_acc_50, 1.0), f"Expected 1.0, got {sel_acc_50}"

    # Selective accuracy at coverage 1.0:
    # All 6 samples -> 4/6 = 2/3
    sel_acc_100 = selective_accuracy(y_true, y_pred, conf, coverage=1.0)
    assert np.isclose(sel_acc_100, 4.0 / 6.0), f"Expected 2/3, got {sel_acc_100}"

    # 3. Risk-coverage curve across [0.5, 1.0]:
    # Risk at 0.5 = 1 - 1.0 = 0.0
    # Risk at 1.0 = 1 - 4/6 = 2/6 = 1/3
    # Trapezoid area = 0.5 * (0.0 + 1/3) / 2 = 1/12 ≈ 0.0833333
    covs, risks, aurc = compute_risk_coverage_curve(y_true, y_pred, conf, coverages=[0.5, 1.0])
    expected_aurc = 0.5 * (0.0 + 1.0 / 3.0) / 2.0
    assert np.isclose(aurc, expected_aurc, atol=1e-5), f"Expected AURC {expected_aurc}, got {aurc}"

    # 4. Domain gap
    gap = compute_domain_gap(0.85, 0.65)
    assert np.isclose(gap, 0.20)
