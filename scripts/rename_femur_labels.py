#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def rename_one(root: Path, src_name: str, dst_name: str) -> None:
    for src_path in root.rglob(src_name):
        if not src_path.is_file():
            continue
        dst_path = src_path.with_name(dst_name)
        if dst_path.exists():
            print(f"skip (exists): {dst_path}")
            continue
        shutil.move(str(src_path), str(dst_path))
        print(f"renamed: {src_path} -> {dst_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename femur label files under a dataset root.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".data/SlicerScenes1"),
        help="Root directory to search (default: .data/SlicerScenes1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    rename_one(root, "femur_left.nii.gz-Necro_L-label.nii.gz", "label_femoral_head_left.nii.gz")
    rename_one(root, "femur_right.nii.gz-Necro_R-label.nii.gz", "label_femoral_head_right.nii.gz")


if __name__ == "__main__":
    main()
