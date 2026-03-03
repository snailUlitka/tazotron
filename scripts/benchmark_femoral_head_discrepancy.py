"""Benchmark discrepancy between manual and heuristic femoral head masks.

Compares manual masks from Slicer scenes with heuristic masks generated in TotalSegmentator case folders.
Default paths:
  - manual: .data/SlicerScenes/<case>/femur_{left,right}.nii.gz-Necro_{L,R}-label.nii.gz
  - heuristic: .data/Totalsegmentator_dataset/<case>/label_femoral_head_{left,right}.nii.gz
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torchio as tio
from tqdm import tqdm

SIDES = ("left", "right")
AFFINE_ATOL = 1e-5
EXPECTED_MASK_DIMS = 3
MANUAL_NAME_CANDIDATES: dict[str, tuple[str, ...]] = {
    "left": ("femur_left.nii.gz-Necro_L-label.nii.gz",),
    "right": ("femur_right.nii.gz-Necro_R-label.nii.gz",),
}
HEURISTIC_NAME_TEMPLATE = "label_femoral_head_{side}.nii.gz"


@dataclass(frozen=True)
class CaseSidePair:
    case_id: str
    side: str
    manual_path: Path
    heuristic_path: Path


@dataclass(frozen=True)
class Counters:
    total_manual_cases: int
    total_heuristic_cases: int
    intersected_cases: int
    evaluated_pairs: int
    missing_manual_pairs: int
    missing_heuristic_pairs: int
    shape_mismatch_pairs: int
    affine_mismatch_pairs: int


@dataclass(frozen=True)
class BenchmarkConfig:
    manual_root: Path
    heuristic_root: Path
    output_csv: Path
    limit_cases: int | None
    case_ids: set[str] | None
    worst_k: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual-root",
        type=Path,
        default=Path(".data/SlicerScenes"),
        help="Root with manual Slicer case folders (default: .data/SlicerScenes).",
    )
    parser.add_argument(
        "--heuristic-root",
        type=Path,
        default=Path(".data/Totalsegmentator_dataset"),
        help="Root with heuristic case folders (default: .data/Totalsegmentator_dataset).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/femoral_head_discrepancy_benchmark.csv"),
        help="Per-side benchmark report path.",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="Evaluate only first N matched cases (for smoke run).",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Evaluate only selected case id(s), e.g. --case-id s0006 --case-id s0052.",
    )
    parser.add_argument(
        "--worst-k",
        type=int,
        default=10,
        help="How many worst rows by Dice to print in summary.",
    )
    return parser


def _is_case_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("s")


def _collect_case_dirs(root: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted(root.iterdir()) if _is_case_dir(path)}


def _resolve_manual_path(case_dir: Path, side: str) -> Path | None:
    for filename in MANUAL_NAME_CANDIDATES[side]:
        candidate = case_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _collect_pairs(
    *,
    manual_root: Path,
    heuristic_root: Path,
    selected_case_ids: set[str] | None,
    limit_cases: int | None,
) -> tuple[list[CaseSidePair], Counters]:
    if not manual_root.is_dir():
        msg = f"manual root is not a directory: {manual_root}"
        raise ValueError(msg)
    if not heuristic_root.is_dir():
        msg = f"heuristic root is not a directory: {heuristic_root}"
        raise ValueError(msg)

    manual_cases = _collect_case_dirs(manual_root)
    heuristic_cases = _collect_case_dirs(heuristic_root)

    case_ids = sorted(set(manual_cases) & set(heuristic_cases))
    if selected_case_ids:
        case_ids = [case_id for case_id in case_ids if case_id in selected_case_ids]
    if limit_cases is not None:
        case_ids = case_ids[:limit_cases]

    pairs: list[CaseSidePair] = []
    missing_manual = 0
    missing_heuristic = 0
    for case_id in case_ids:
        manual_case = manual_cases[case_id]
        heuristic_case = heuristic_cases[case_id]
        for side in SIDES:
            manual_path = _resolve_manual_path(manual_case, side)
            heuristic_path = heuristic_case / HEURISTIC_NAME_TEMPLATE.format(side=side)
            if manual_path is None:
                missing_manual += 1
                continue
            if not heuristic_path.is_file():
                missing_heuristic += 1
                continue
            pairs.append(
                CaseSidePair(
                    case_id=case_id,
                    side=side,
                    manual_path=manual_path,
                    heuristic_path=heuristic_path,
                ),
            )

    counters = Counters(
        total_manual_cases=len(manual_cases),
        total_heuristic_cases=len(heuristic_cases),
        intersected_cases=len(case_ids),
        evaluated_pairs=len(pairs),
        missing_manual_pairs=missing_manual,
        missing_heuristic_pairs=missing_heuristic,
        shape_mismatch_pairs=0,
        affine_mismatch_pairs=0,
    )
    return pairs, counters


def _load_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = tio.LabelMap(path)
    array = image.data.detach().cpu().numpy()
    mask = np.squeeze(array) > 0
    if mask.ndim != EXPECTED_MASK_DIMS:
        msg = f"Expected 3D mask in {path}, got shape {array.shape}"
        raise ValueError(msg)
    return mask, np.asarray(image.affine, dtype=np.float64)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def _segmentation_metrics(*, gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())

    gt_voxels = int(gt.sum())
    pred_voxels = int(pred.sum())
    union = tp + fp + fn

    if union == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = _safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
        iou = _safe_divide(tp, union)

    precision = (1.0 if gt_voxels == 0 else 0.0) if tp + fp == 0 else _safe_divide(tp, tp + fp)

    recall = (1.0 if pred_voxels == 0 else 0.0) if tp + fn == 0 else _safe_divide(tp, tp + fn)

    abs_volume_diff_vox = abs(pred_voxels - gt_voxels)
    rel_volume_diff = (
        0.0 if gt_voxels == 0 and pred_voxels == 0 else _safe_divide(abs_volume_diff_vox, float(gt_voxels))
    )

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "manual_voxels": float(gt_voxels),
        "heuristic_voxels": float(pred_voxels),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "abs_volume_diff_vox": float(abs_volume_diff_vox),
        "rel_volume_diff": float(rel_volume_diff),
    }


def _describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _format_stats(name: str, values: list[float]) -> str:
    stats = _describe(values)
    return (
        f"{name}: "
        f"mean={stats['mean']:.4f}, "
        f"median={stats['median']:.4f}, "
        f"p05={stats['p05']:.4f}, "
        f"p95={stats['p95']:.4f}, "
        f"min={stats['min']:.4f}, "
        f"max={stats['max']:.4f}"
    )


def _write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        msg = "No rows to write to CSV."
        raise ValueError(msg)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, float | int | str]], counters: Counters, worst_k: int) -> None:
    print("summary:")
    print(f"  manual_cases_total={counters.total_manual_cases}")
    print(f"  heuristic_cases_total={counters.total_heuristic_cases}")
    print(f"  intersected_cases={counters.intersected_cases}")
    print(f"  evaluated_pairs={counters.evaluated_pairs}")
    print(f"  missing_manual_pairs={counters.missing_manual_pairs}")
    print(f"  missing_heuristic_pairs={counters.missing_heuristic_pairs}")
    print(f"  shape_mismatch_pairs={counters.shape_mismatch_pairs}")
    print(f"  affine_mismatch_pairs={counters.affine_mismatch_pairs}")

    all_dice = [float(row["dice"]) for row in rows]
    all_iou = [float(row["iou"]) for row in rows]
    all_precision = [float(row["precision"]) for row in rows]
    all_recall = [float(row["recall"]) for row in rows]
    all_rel_diff = [float(row["rel_volume_diff"]) for row in rows if not math.isnan(float(row["rel_volume_diff"]))]

    print(_format_stats("  dice", all_dice))
    print(_format_stats("  iou", all_iou))
    print(_format_stats("  precision", all_precision))
    print(_format_stats("  recall", all_recall))
    print(_format_stats("  rel_volume_diff", all_rel_diff))

    for side in SIDES:
        side_rows = [row for row in rows if row["side"] == side]
        side_dice = [float(row["dice"]) for row in side_rows]
        print(_format_stats(f"  {side}_dice", side_dice))

    worst_rows = sorted(rows, key=lambda row: float(row["dice"]))[: max(0, worst_k)]
    if worst_rows:
        print(f"  worst_{len(worst_rows)}_by_dice:")
        for row in worst_rows:
            print(
                f"    {row['case_id']}:{row['side']} "
                f"dice={float(row['dice']):.4f}, "
                f"iou={float(row['iou']):.4f}, "
                f"rel_volume_diff={float(row['rel_volume_diff']):.4f}",
            )


def benchmark(config: BenchmarkConfig) -> list[dict[str, float | int | str]]:
    pairs, counters = _collect_pairs(
        manual_root=config.manual_root,
        heuristic_root=config.heuristic_root,
        selected_case_ids=config.case_ids,
        limit_cases=config.limit_cases,
    )
    if not pairs:
        msg = "No valid case-side pairs found to evaluate."
        raise ValueError(msg)

    rows: list[dict[str, float | int | str]] = []
    shape_mismatch_pairs = 0
    affine_mismatch_pairs = 0
    for pair in tqdm(pairs, desc="Benchmarking", mininterval=1.0):
        manual_mask, manual_affine = _load_mask(pair.manual_path)
        heuristic_mask, heuristic_affine = _load_mask(pair.heuristic_path)

        if manual_mask.shape != heuristic_mask.shape:
            shape_mismatch_pairs += 1
            print(
                f"skip shape mismatch {pair.case_id}:{pair.side} "
                f"manual={manual_mask.shape}, heuristic={heuristic_mask.shape}",
            )
            continue

        affine_match = bool(np.allclose(manual_affine, heuristic_affine, atol=AFFINE_ATOL))
        if not affine_match:
            affine_mismatch_pairs += 1

        metrics = _segmentation_metrics(gt=manual_mask, pred=heuristic_mask)
        rows.append(
            {
                "case_id": pair.case_id,
                "side": pair.side,
                "manual_path": str(pair.manual_path),
                "heuristic_path": str(pair.heuristic_path),
                "shape_z": int(manual_mask.shape[0]),
                "shape_y": int(manual_mask.shape[1]),
                "shape_x": int(manual_mask.shape[2]),
                "affine_match": int(affine_match),
                **metrics,
            },
        )

    updated_counters = Counters(
        total_manual_cases=counters.total_manual_cases,
        total_heuristic_cases=counters.total_heuristic_cases,
        intersected_cases=counters.intersected_cases,
        evaluated_pairs=len(rows),
        missing_manual_pairs=counters.missing_manual_pairs,
        missing_heuristic_pairs=counters.missing_heuristic_pairs,
        shape_mismatch_pairs=shape_mismatch_pairs,
        affine_mismatch_pairs=affine_mismatch_pairs,
    )

    if not rows:
        msg = "No rows produced after filtering (shape mismatches or missing files)."
        raise ValueError(msg)

    _write_csv(rows, config.output_csv)
    _print_summary(rows, updated_counters, config.worst_k)
    print(f"saved_csv={config.output_csv}")
    return rows


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    case_ids = set(args.case_id) if args.case_id else None
    config = BenchmarkConfig(
        manual_root=args.manual_root,
        heuristic_root=args.heuristic_root,
        output_csv=args.output_csv,
        limit_cases=args.limit_cases,
        case_ids=case_ids,
        worst_k=args.worst_k,
    )
    benchmark(config)


if __name__ == "__main__":
    main()
