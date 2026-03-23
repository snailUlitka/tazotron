"""Helpers for before/after X-ray pose benchmarks."""

from __future__ import annotations

import csv
import json
import math
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image

from tazotron.xray_generation import RenderBenchmarkRow, squeeze_xray_tensor

if TYPE_CHECKING:
    from pathlib import Path

MIN_FOLDS = 2
IMAGE_SCALE_EPS = 1e-6


def build_case_level_split_rows(
    case_ids: list[str],
    *,
    test_ratio: float = 0.10,
    folds: int = 5,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Create a deterministic case-level hold-out + k-fold split manifest."""
    if not case_ids:
        msg = "case_ids must not be empty."
        raise ValueError(msg)
    if not 0.0 < test_ratio < 1.0:
        msg = f"test_ratio must be in (0, 1), got {test_ratio}"
        raise ValueError(msg)
    if folds < MIN_FOLDS:
        msg = "folds must be at least 2."
        raise ValueError(msg)

    shuffled = sorted(case_ids)
    random.Random(seed).shuffle(shuffled)  # noqa: S311
    if len(shuffled) == 1:
        return [{"case_id": shuffled[0], "split": "test", "fold": ""}]

    test_size = max(1, min(len(shuffled) - 1, round(len(shuffled) * test_ratio)))
    test_cases = sorted(shuffled[:test_size])
    trainval_cases = shuffled[test_size:]
    effective_folds = min(folds, len(trainval_cases))
    rows: list[dict[str, object]] = [{"case_id": case_id, "split": "test", "fold": ""} for case_id in test_cases]

    for index, case_id in enumerate(trainval_cases):
        rows.append({"case_id": case_id, "split": "trainval", "fold": index % effective_folds})
    return sorted(rows, key=lambda row: str(row["case_id"]))


def summarize_render_rows(rows: list[RenderBenchmarkRow]) -> dict[str, float | int]:
    """Aggregate per-case render metrics into summary statistics."""
    if not rows:
        return {
            "total_cases": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "empty_count": 0,
            "empty_rate": 0.0,
            "invalid_mask_count": 0,
            "invalid_mask_rate": 0.0,
            "center_offset_abs_mean_px": math.nan,
            "center_offset_abs_median_px": math.nan,
            "min_edge_margin_mean_px": math.nan,
            "min_edge_margin_median_px": math.nan,
            "projected_area_ratio_mean": math.nan,
            "projected_area_ratio_median": math.nan,
            "border_touch_count": 0,
        }

    success_rows = [row for row in rows if row.status == "success"]
    center_offsets = [
        math.hypot(float(row.bbox_center_offset_x_px), float(row.bbox_center_offset_y_px))
        for row in success_rows
        if row.bbox_center_offset_x_px is not None and row.bbox_center_offset_y_px is not None
    ]
    edge_margins = [float(row.min_edge_margin_px) for row in success_rows if row.min_edge_margin_px is not None]
    area_ratios = [float(row.projected_area_ratio) for row in success_rows if row.projected_area_ratio is not None]

    return {
        "total_cases": len(rows),
        "success_count": len(success_rows),
        "success_rate": len(success_rows) / len(rows),
        "empty_count": sum(1 for row in rows if row.reason == "empty_drr"),
        "empty_rate": sum(1 for row in rows if row.reason == "empty_drr") / len(rows),
        "invalid_mask_count": sum(1 for row in rows if row.reason == "invalid_bilateral_mask"),
        "invalid_mask_rate": sum(1 for row in rows if row.reason == "invalid_bilateral_mask") / len(rows),
        "center_offset_abs_mean_px": _safe_mean(center_offsets),
        "center_offset_abs_median_px": _safe_median(center_offsets),
        "min_edge_margin_mean_px": _safe_mean(edge_margins),
        "min_edge_margin_median_px": _safe_median(edge_margins),
        "projected_area_ratio_mean": _safe_mean(area_ratios),
        "projected_area_ratio_median": _safe_median(area_ratios),
        "border_touch_count": sum(int(row.border_touch or 0) for row in success_rows),
    }


def compare_render_reports(
    before_rows: list[RenderBenchmarkRow],
    after_rows: list[RenderBenchmarkRow],
) -> tuple[list[dict[str, object]], dict[str, float | int]]:
    """Join before/after render metrics on case id and compute deltas."""
    before_map = {row.case_id: row for row in before_rows}
    after_map = {row.case_id: row for row in after_rows}
    common_case_ids = sorted(set(before_map) & set(after_map))
    comparison_rows: list[dict[str, object]] = []

    for case_id in common_case_ids:
        before = before_map[case_id]
        after = after_map[case_id]
        comparison_rows.append(
            {
                "case_id": case_id,
                "before_status": before.status,
                "before_reason": before.reason,
                "after_status": after.status,
                "after_reason": after.reason,
                "before_center_offset_abs_px": _center_offset_abs(before),
                "after_center_offset_abs_px": _center_offset_abs(after),
                "delta_center_offset_abs_px": _delta(_center_offset_abs(after), _center_offset_abs(before)),
                "before_min_edge_margin_px": before.min_edge_margin_px,
                "after_min_edge_margin_px": after.min_edge_margin_px,
                "delta_min_edge_margin_px": _delta(after.min_edge_margin_px, before.min_edge_margin_px),
                "before_projected_area_ratio": before.projected_area_ratio,
                "after_projected_area_ratio": after.projected_area_ratio,
                "delta_projected_area_ratio": _delta(after.projected_area_ratio, before.projected_area_ratio),
                "before_border_touch": before.border_touch,
                "after_border_touch": after.border_touch,
            }
        )

    before_success = [row for row in comparison_rows if row["before_status"] == "success"]
    after_success = [row for row in comparison_rows if row["after_status"] == "success"]
    paired_success = [
        row for row in comparison_rows if row["before_status"] == "success" and row["after_status"] == "success"
    ]
    before_center_offsets = [
        float(row["before_center_offset_abs_px"])
        for row in paired_success
        if row["before_center_offset_abs_px"] is not None
    ]
    after_center_offsets = [
        float(row["after_center_offset_abs_px"])
        for row in paired_success
        if row["after_center_offset_abs_px"] is not None
    ]
    before_edge_margins = [
        float(row["before_min_edge_margin_px"])
        for row in paired_success
        if row["before_min_edge_margin_px"] is not None
    ]
    after_edge_margins = [
        float(row["after_min_edge_margin_px"]) for row in paired_success if row["after_min_edge_margin_px"] is not None
    ]
    before_area_ratios = [
        float(row["before_projected_area_ratio"])
        for row in paired_success
        if row["before_projected_area_ratio"] is not None
    ]
    after_area_ratios = [
        float(row["after_projected_area_ratio"])
        for row in paired_success
        if row["after_projected_area_ratio"] is not None
    ]
    summary = {
        "eligible_case_count": len(common_case_ids),
        "paired_success_count": len(paired_success),
        "before_success_rate": len(before_success) / len(common_case_ids) if common_case_ids else 0.0,
        "after_success_rate": len(after_success) / len(common_case_ids) if common_case_ids else 0.0,
        "before_empty_count": sum(1 for row in comparison_rows if row["before_reason"] == "empty_drr"),
        "after_empty_count": sum(1 for row in comparison_rows if row["after_reason"] == "empty_drr"),
        "before_invalid_mask_count": sum(
            1 for row in comparison_rows if row["before_reason"] == "invalid_bilateral_mask"
        ),
        "after_invalid_mask_count": sum(
            1 for row in comparison_rows if row["after_reason"] == "invalid_bilateral_mask"
        ),
        "before_center_offset_abs_mean_px": _safe_mean(before_center_offsets),
        "after_center_offset_abs_mean_px": _safe_mean(after_center_offsets),
        "before_center_offset_abs_median_px": _safe_median(before_center_offsets),
        "after_center_offset_abs_median_px": _safe_median(after_center_offsets),
        "before_min_edge_margin_mean_px": _safe_mean(before_edge_margins),
        "after_min_edge_margin_mean_px": _safe_mean(after_edge_margins),
        "before_min_edge_margin_median_px": _safe_median(before_edge_margins),
        "after_min_edge_margin_median_px": _safe_median(after_edge_margins),
        "before_projected_area_ratio_mean": _safe_mean(before_area_ratios),
        "after_projected_area_ratio_mean": _safe_mean(after_area_ratios),
        "before_projected_area_ratio_median": _safe_median(before_area_ratios),
        "after_projected_area_ratio_median": _safe_median(after_area_ratios),
        "before_border_touch_count": sum(int(row["before_border_touch"] or 0) for row in paired_success),
        "after_border_touch_count": sum(int(row["after_border_touch"] or 0) for row in paired_success),
        "delta_border_touch_count": sum(int(row["after_border_touch"] or 0) for row in paired_success)
        - sum(int(row["before_border_touch"] or 0) for row in paired_success),
    }
    return comparison_rows, summary


def tensor_to_grayscale_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a DRR tensor to a grayscale image."""
    image = squeeze_xray_tensor(tensor)
    min_value = float(image.min().item())
    max_value = float(image.max().item())
    scaled = (
        torch.zeros_like(image)
        if max_value - min_value < IMAGE_SCALE_EPS
        else (image - min_value) / (max_value - min_value)
    )
    array = (scaled * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode="L")


def diff_to_rgb_image(before: torch.Tensor, after: torch.Tensor) -> Image.Image:
    """Convert a before/after DRR difference into a red/white/blue image."""
    before_image = squeeze_xray_tensor(before)
    after_image = squeeze_xray_tensor(after)
    diff = after_image - before_image
    vmax = float(diff.abs().max().item()) if diff.numel() else 0.0
    vmax = max(vmax, 1e-6)
    norm = (diff / vmax).clamp(-1.0, 1.0)
    base = 1.0 - norm.abs()
    red = norm.clamp(min=0.0)
    blue = (-norm).clamp(min=0.0)
    rgb = torch.stack((base + red, base, base + blue), dim=-1).clamp(0.0, 1.0)
    array = (rgb * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Persist a list of dict rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else math.nan


def _safe_median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else math.nan


def _center_offset_abs(row: RenderBenchmarkRow) -> float | None:
    if row.bbox_center_offset_x_px is None or row.bbox_center_offset_y_px is None:
        return None
    return math.hypot(float(row.bbox_center_offset_x_px), float(row.bbox_center_offset_y_px))


def _delta(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs) - float(rhs)
