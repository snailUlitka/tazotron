"""Build train/val CSV manifests for binary X-ray classification."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

HEALTHY_DIRNAME = "without_necro"
NECROSIS_DIRNAME = "with_necro"
HEALTHY_LABEL = "healthy"
NECROSIS_LABEL = "necrosis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val CSV manifests from paired .pt X-ray tensors.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".data/output_with_crop"),
        help="Directory containing with_necro/ and without_necro/ folders.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Output path for train.csv (default: <data-root>/train.csv).",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Output path for val.csv (default: <data-root>/val.csv).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of paired case ids to place into validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic case-id splitting.",
    )
    return parser.parse_args()


def collect_case_ids(data_root: Path) -> list[str]:
    with_necro_dir = data_root / NECROSIS_DIRNAME
    without_necro_dir = data_root / HEALTHY_DIRNAME

    if not with_necro_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {with_necro_dir}")
    if not without_necro_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {without_necro_dir}")

    with_necro_ids = {path.stem for path in with_necro_dir.glob("*.pt")}
    without_necro_ids = {path.stem for path in without_necro_dir.glob("*.pt")}
    paired_ids = sorted(with_necro_ids & without_necro_ids)

    if not paired_ids:
        raise ValueError(f"No paired .pt files found in {data_root}")

    return paired_ids


def split_case_ids(case_ids: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1:
        raise ValueError("At least two paired case ids are required to build train/val splits.")

    val_size = max(1, min(len(shuffled) - 1, round(len(shuffled) * val_fraction)))
    val_ids = sorted(shuffled[:val_size])
    train_ids = sorted(shuffled[val_size:])
    return train_ids, val_ids


def build_rows(case_ids: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_id in case_ids:
        rows.append({"path": f"{HEALTHY_DIRNAME}/{case_id}.pt", "label": HEALTHY_LABEL})
        rows.append({"path": f"{NECROSIS_DIRNAME}/{case_id}.pt", "label": NECROSIS_LABEL})
    return rows


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    train_csv = args.train_csv.resolve() if args.train_csv is not None else data_root / "train.csv"
    val_csv = args.val_csv.resolve() if args.val_csv is not None else data_root / "val.csv"

    case_ids = collect_case_ids(data_root)
    train_ids, val_ids = split_case_ids(case_ids, args.val_fraction, args.seed)

    train_rows = build_rows(train_ids)
    val_rows = build_rows(val_ids)

    write_rows(train_csv, train_rows)
    write_rows(val_csv, val_rows)

    print(f"Paired cases: {len(case_ids)}")
    print(f"Train cases: {len(train_ids)} -> rows: {len(train_rows)}")
    print(f"Val cases: {len(val_ids)} -> rows: {len(val_rows)}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")


if __name__ == "__main__":
    main()
