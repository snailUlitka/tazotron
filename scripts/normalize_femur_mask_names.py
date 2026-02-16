"""Normalize femur mask filenames to include .nii.gz.seg.nrrd suffix."""

from __future__ import annotations

import argparse
from pathlib import Path

LEFT_CANON = "femur_left.nii.gz.seg.nrrd"
RIGHT_CANON = "femur_right.nii.gz.seg.nrrd"
LEFT_ALTS = ("femur_left.nii.seg.nrrd", "femur_left.nii.gz.nii.seg.nrrd")
RIGHT_ALTS = ("femur_right.nii.seg.nrrd", "femur_right.nii.gz.nii.seg.nrrd")


def _normalize_one(
    case_dir: Path,
    *,
    canon_name: str,
    alt_names: tuple[str, ...],
    dry_run: bool,
) -> tuple[int, int, int]:
    canon_path = case_dir / canon_name
    alt_paths = [case_dir / name for name in alt_names]
    existing_alts = [path for path in alt_paths if path.exists()]

    if canon_path.exists():
        conflicts = len(existing_alts)
        for alt_path in existing_alts:
            print(f"conflict: {alt_path} and {canon_path} both exist")
        return 0, 1, conflicts

    if not existing_alts:
        return 0, 1, 0

    if len(existing_alts) > 1:
        for alt_path in existing_alts:
            print(f"conflict: multiple alternatives found for {canon_path}: {alt_path}")
        return 0, 0, len(existing_alts)

    src_path = existing_alts[0]
    if dry_run:
        print(f"[dry-run] {src_path} -> {canon_path}")
    else:
        src_path.rename(canon_path)
        print(f"renamed {src_path} -> {canon_path}")
    return 1, 0, 0


def _normalize_case(case_dir: Path, *, dry_run: bool) -> tuple[int, int, int]:
    renamed_left, skipped_left, conflicts_left = _normalize_one(
        case_dir,
        canon_name=LEFT_CANON,
        alt_names=LEFT_ALTS,
        dry_run=dry_run,
    )
    renamed_right, skipped_right, conflicts_right = _normalize_one(
        case_dir,
        canon_name=RIGHT_CANON,
        alt_names=RIGHT_ALTS,
        dry_run=dry_run,
    )
    return (
        renamed_left + renamed_right,
        skipped_left + skipped_right,
        conflicts_left + conflicts_right,
    )


def normalize(root: Path, *, dry_run: bool) -> int:
    if not root.is_dir():
        msg = f"{root} is not a directory"
        raise ValueError(msg)

    total_renamed = 0
    total_skipped = 0
    total_conflicts = 0

    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        renamed, skipped, conflicts = _normalize_case(case_dir, dry_run=dry_run)
        total_renamed += renamed
        total_skipped += skipped
        total_conflicts += conflicts

    print(
        "done: "
        f"renamed={total_renamed}, "
        f"skipped={total_skipped}, "
        f"conflicts={total_conflicts}",
    )
    return total_renamed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing case subfolders.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned renames.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    normalize(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
