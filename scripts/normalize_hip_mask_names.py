"""Normalize hip mask filenames to include .nii.gz.seg.nrrd suffix."""

from __future__ import annotations

import argparse
from pathlib import Path


LEFT_CANON = "hip_left.nii.gz.seg.nrrd"
RIGHT_CANON = "hip_right.nii.gz.seg.nrrd"
LEFT_ALT = "hip_left.nii.seg.nrrd"
RIGHT_ALT = "hip_right.nii.seg.nrrd"


def _normalize_case(case_dir: Path, *, dry_run: bool) -> tuple[int, int, int]:
    renamed = 0
    skipped = 0
    conflicts = 0

    for alt_name, canon_name in ((LEFT_ALT, LEFT_CANON), (RIGHT_ALT, RIGHT_CANON)):
        alt_path = case_dir / alt_name
        canon_path = case_dir / canon_name
        if canon_path.exists():
            if alt_path.exists():
                conflicts += 1
                print(f"conflict: {alt_path} and {canon_path} both exist")
            skipped += 1
            continue
        if not alt_path.exists():
            skipped += 1
            continue
        if dry_run:
            print(f"[dry-run] {alt_path} -> {canon_path}")
        else:
            alt_path.rename(canon_path)
            print(f"renamed {alt_path} -> {canon_path}")
        renamed += 1

    return renamed, skipped, conflicts


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
