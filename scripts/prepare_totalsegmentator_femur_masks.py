"""Prepare TotalSegmentator case folders for `tazotron add_femoral_head_masks`.

This script converts femur masks from:
  - segmentations/femur_left.nii.gz
  - segmentations/femur_right.nii.gz

into canonical filenames expected by the current CTDataset defaults:
  - femur_left.nii.gz.seg.nrrd
  - femur_right.nii.gz.seg.nrrd
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import SimpleITK

LEFT_SRC = Path("segmentations/femur_left.nii.gz")
RIGHT_SRC = Path("segmentations/femur_right.nii.gz")
LEFT_DST = Path("femur_left.nii.gz.seg.nrrd")
RIGHT_DST = Path("femur_right.nii.gz.seg.nrrd")


@dataclass
class Counters:
    planned: int = 0
    converted: int = 0
    skipped_existing: int = 0
    missing_source: int = 0
    errors: int = 0


def _convert_one(
    case_dir: Path,
    *,
    src_rel: Path,
    dst_rel: Path,
    overwrite: bool,
    dry_run: bool,
) -> Counters:
    src_path = case_dir / src_rel
    dst_path = case_dir / dst_rel
    counters = Counters()

    if not src_path.is_file():
        counters.missing_source = 1
        print(f"missing: {src_path}")
        return counters

    if dst_path.exists() and not overwrite:
        counters.skipped_existing = 1
        return counters

    if dry_run:
        action = "overwrite" if dst_path.exists() else "convert"
        print(f"[dry-run] {action}: {src_path} -> {dst_path}")
        counters.planned = 1
        return counters

    try:
        image = SimpleITK.ReadImage(str(src_path))
        SimpleITK.WriteImage(image, str(dst_path), useCompression=True)
        print(f"converted: {src_path} -> {dst_path}")
        counters.converted = 1
    except Exception as exc:
        print(f"error: {src_path} -> {dst_path}: {exc}")
        counters.errors = 1
    return counters


def _add_into(total: Counters, delta: Counters) -> None:
    total.planned += delta.planned
    total.converted += delta.converted
    total.skipped_existing += delta.skipped_existing
    total.missing_source += delta.missing_source
    total.errors += delta.errors


def prepare(root: Path, *, overwrite: bool, dry_run: bool) -> Counters:
    if not root.is_dir():
        msg = f"{root} is not a directory"
        raise ValueError(msg)

    total = Counters()

    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        left = _convert_one(
            case_dir,
            src_rel=LEFT_SRC,
            dst_rel=LEFT_DST,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        right = _convert_one(
            case_dir,
            src_rel=RIGHT_SRC,
            dst_rel=RIGHT_DST,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        _add_into(total, left)
        _add_into(total, right)

    print(
        "done: "
        f"planned={total.planned}, "
        f"converted={total.converted}, "
        f"skipped_existing={total.skipped_existing}, "
        f"missing_source={total.missing_source}, "
        f"errors={total.errors}",
    )
    return total


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        default=Path(".data/Totalsegmentator_dataset"),
        nargs="?",
        help="Directory containing case subfolders (default: .data/Totalsegmentator_dataset).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing femur_*.nii.gz.seg.nrrd files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned actions.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    prepare(args.root, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
