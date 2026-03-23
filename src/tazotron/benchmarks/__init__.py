"""Benchmark helpers for X-ray render/training comparisons."""

from tazotron.benchmarks.xray_pose import (
    build_case_level_split_rows,
    compare_render_reports,
    diff_to_rgb_image,
    summarize_render_rows,
    tensor_to_grayscale_image,
    write_csv_rows,
    write_json,
)

__all__ = [
    "build_case_level_split_rows",
    "compare_render_reports",
    "diff_to_rgb_image",
    "summarize_render_rows",
    "tensor_to_grayscale_image",
    "write_csv_rows",
    "write_json",
]
