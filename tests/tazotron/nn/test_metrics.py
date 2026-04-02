"""Tests for shared binary classification metrics."""

from __future__ import annotations

import math

from tazotron.nn.metrics import binary_roc_auc, compute_binary_metrics, metric_for_selection, safe_divide


def test_safe_divide_returns_zero_for_zero_denominator() -> None:
    assert safe_divide(5.0, 0.0) == 0.0


def test_binary_roc_auc_handles_ties() -> None:
    auc = binary_roc_auc([0, 1, 0, 1], [0.1, 0.8, 0.8, 0.8])
    assert auc == 0.75


def test_compute_binary_metrics_matches_expected_values() -> None:
    metrics = compute_binary_metrics(
        [0, 1, 1, 0],
        [0.1, 0.7, 0.4, 0.8],
        loss=0.5,
    )

    assert metrics == {
        "loss": 0.5,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "roc_auc": 0.5,
    }


def test_metric_for_selection_maps_nan_to_negative_infinity() -> None:
    score = metric_for_selection({"roc_auc": float("nan")}, "roc_auc")
    assert math.isinf(score)
    assert score < 0
