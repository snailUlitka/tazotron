"""Binary classification metrics shared across training notebooks."""

from __future__ import annotations

import math

import torch


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two floats and return 0 when the denominator is 0."""
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_roc_auc(targets: list[int], scores: list[float]) -> float:
    """Compute binary ROC-AUC directly from scores without sklearn."""
    targets_tensor = torch.as_tensor(targets, dtype=torch.long)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
    positive_mask = targets_tensor == 1
    negative_mask = targets_tensor == 0
    n_positive = int(positive_mask.sum().item())
    n_negative = int(negative_mask.sum().item())
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = torch.argsort(scores_tensor)
    sorted_scores = scores_tensor[order]
    ranks = torch.zeros_like(scores_tensor, dtype=torch.float32)

    start = 0
    current_rank = 1.0
    while start < len(sorted_scores):
        end = start
        while end + 1 < len(sorted_scores) and torch.isclose(sorted_scores[end + 1], sorted_scores[start]).item():
            end += 1
        average_rank = (current_rank + current_rank + (end - start)) / 2.0
        ranks[order[start : end + 1]] = average_rank
        current_rank += end - start + 1
        start = end + 1

    positive_rank_sum = float(ranks[positive_mask].sum().item())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def compute_binary_metrics(targets: list[int], scores: list[float], *, loss: float) -> dict[str, float]:
    """Compute loss and threshold-based binary metrics from scores."""
    targets_tensor = torch.as_tensor(targets, dtype=torch.long)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
    predictions = (scores_tensor >= 0.5).long()

    tp = int(((predictions == 1) & (targets_tensor == 1)).sum().item())
    tn = int(((predictions == 0) & (targets_tensor == 0)).sum().item())
    fp = int(((predictions == 1) & (targets_tensor == 0)).sum().item())
    fn = int(((predictions == 0) & (targets_tensor == 1)).sum().item())

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    return {
        "loss": float(loss),
        "accuracy": float(safe_divide(tp + tn, len(targets))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(safe_divide(2 * precision * recall, precision + recall)),
        "roc_auc": float(binary_roc_auc(targets, scores)),
    }


def metric_for_selection(metrics: dict[str, float], metric_name: str) -> float:
    """Convert a metric into a selection-ready score, treating non-finite values as -inf."""
    value = float(metrics[metric_name])
    return value if math.isfinite(value) else float("-inf")

