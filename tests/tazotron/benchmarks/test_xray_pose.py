import pytest
import torch

from tazotron.benchmarks.xray_pose import build_case_level_split_rows, compare_render_reports
from tazotron.xray_generation import RenderBenchmarkRow, compute_projection_bbox_metrics

EXPECTED_CASE_COUNT = 5
EXPECTED_JOINED_CASES = 2
EXPECTED_BEFORE_SUCCESS_RATE = 0.5
EXPECTED_DELTA_MARGIN = 4.0
EXPECTED_BBOX_WIDTH = 3
EXPECTED_BBOX_HEIGHT = 4


@pytest.mark.fast
def test_build_case_level_split_rows_keeps_cases_unique() -> None:
    rows = build_case_level_split_rows(["s0001", "s0002", "s0003", "s0004", "s0005"], test_ratio=0.2, folds=3, seed=42)

    assert len(rows) == EXPECTED_CASE_COUNT
    assert {row["case_id"] for row in rows} == {"s0001", "s0002", "s0003", "s0004", "s0005"}
    assert sum(1 for row in rows if row["split"] == "test") == 1
    assert all(row["split"] == "test" or isinstance(row["fold"], int) for row in rows)


@pytest.mark.fast
def test_compare_render_reports_tracks_success_rates_and_deltas() -> None:
    before_rows = [
        RenderBenchmarkRow(
            "s0001", "legacy_crop", "success", "", 800.0, 1020.0, 2.0, 2.0, 200, 200, 50, 60, 1.0, -2.0, 4.0, 0.2, 0
        ),
        RenderBenchmarkRow(
            "s0002",
            "legacy_crop",
            "skipped",
            "empty_drr",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    ]
    after_rows = [
        RenderBenchmarkRow(
            "s0001", "autopose", "success", "", 800.0, 1020.0, 1.8, 1.8, 200, 200, 70, 80, 0.5, -1.0, 8.0, 0.3, 0
        ),
        RenderBenchmarkRow(
            "s0002", "autopose", "success", "", 800.0, 1020.0, 1.8, 1.8, 200, 200, 60, 60, 0.0, 0.0, 5.0, 0.25, 0
        ),
    ]

    rows, summary = compare_render_reports(before_rows, after_rows)

    assert len(rows) == EXPECTED_JOINED_CASES
    assert summary["eligible_case_count"] == EXPECTED_JOINED_CASES
    assert summary["before_success_rate"] == EXPECTED_BEFORE_SUCCESS_RATE
    assert summary["after_success_rate"] == 1.0
    first = next(row for row in rows if row["case_id"] == "s0001")
    assert first["delta_min_edge_margin_px"] == EXPECTED_DELTA_MARGIN


@pytest.mark.fast
def test_compute_projection_bbox_metrics_detects_border_touch() -> None:
    projection = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
    projection[0, 0, 1:5, 0:3] = 1.0

    metrics = compute_projection_bbox_metrics(projection)

    assert metrics["projected_bbox_width_px"] == EXPECTED_BBOX_WIDTH
    assert metrics["projected_bbox_height_px"] == EXPECTED_BBOX_HEIGHT
    assert metrics["border_touch"] == 1
    assert metrics["min_edge_margin_px"] == 0.0
