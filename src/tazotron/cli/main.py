"""CLI entrypoint for tazotron utilities."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchio as tio
from PIL import Image
from tqdm import tqdm

from tazotron.datasets.transforms import CTToXRTransform

if TYPE_CHECKING:
    from collections.abc import Sequence

DRR_BATCH_RANK = 4


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(prog="tazotron", description="CT-to-DRR tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ct2xr_parser = subparsers.add_parser("ct2xr", help="Render XR/DRR from CT with ROI crop.")
    ct2xr_parser.add_argument("--volume", required=True, help="Path to CT volume (.nii.gz).")
    ct2xr_parser.add_argument(
        "--labels",
        required=True,
        help="Comma-separated labelmap paths (.nii.gz) used for cropping ROI.",
    )
    ct2xr_parser.add_argument("-o", "--output", required=True, help="Output XR path (.tiff).")
    ct2xr_parser.add_argument(
        "--necro",
        type=int,
        default=0,
        help="Inject N random necrosis spots into the CT (0 disables).",
    )

    subparsers.add_parser("build_dataset", help="Build XR dataset from .data/SlicerScenes.")

    args = parser.parse_args(argv)

    if args.command == "ct2xr":
        run_ct2xr(args)
    elif args.command == "build_dataset":
        run_build_dataset()
    else:
        parser.error(f"Unknown command {args.command}")


def run_ct2xr(args: argparse.Namespace) -> None:
    """Execute CT → ROI crop → DRR render and save TIFF."""
    ct_path = Path(args.volume)
    label_paths = [Path(p.strip()) for p in args.labels.split(",") if p.strip()]
    if not label_paths:
        msg = "At least one labelmap path must be provided."
        raise ValueError(msg)

    subject = build_subject(ct_path, label_paths)
    pipeline = CTToXRTransform(necrosis_spots=max(args.necro, 0))

    subject = pipeline(subject)
    drr = subject["drr"].detach().cpu().clamp(0, 1)
    save_tiff(drr, Path(args.output))


def build_subject(ct_path: Path, label_paths: list[Path], label_key: str = "label") -> tio.Subject:
    """Create torchio.Subject from CT and label paths."""
    subject = tio.Subject(volume=tio.ScalarImage(ct_path))
    label = merge_labelmaps(label_paths)
    subject[label_key] = label
    return subject


def merge_labelmaps(label_paths: list[Path]) -> tio.LabelMap:
    """Merge labelmaps and assign distinct IDs to preserve laterality."""
    base = tio.LabelMap(label_paths[0])
    combined = torch.zeros_like(base.data)
    for idx, path in enumerate(label_paths):
        img = tio.LabelMap(path)
        if img.data.shape != combined.shape:
            msg = f"Labelmap shape mismatch for {path}"
            raise ValueError(msg)
        mask = (img.data > 0).to(combined.dtype) * (idx + 1)
        combined = torch.where(mask > 0, mask, combined)
    return tio.LabelMap(tensor=combined, affine=base.affine)


def save_tiff(drr: torch.Tensor, output_path: Path) -> None:
    """Save DRR tensor as 16-bit TIFF (lossless deflate)."""
    if drr.ndim == DRR_BATCH_RANK:
        drr = drr[0, 0]
    array = (drr * 65535).to(torch.uint16).cpu().numpy()
    image = Image.fromarray(array, mode="I;16")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, compression="tiff_deflate")


def run_build_dataset() -> None:
    """Render dataset from .data/SlicerScenes into .data/output with metadata."""
    scenes_root = Path(".data/SlicerScenes")
    output_root = Path(".data/output")
    output_root.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, int] = {}
    index = 0
    flush_every = 5
    written_since_flush = 0
    scene_dirs = sorted([p for p in scenes_root.iterdir() if p.is_dir() and p.name.startswith("s")])
    for scene in tqdm(scene_dirs, desc="Scenes"):
        ct_path = scene / "ct.nii.gz"
        label_paths = [
            scene / "femur_left.nii.gz-Necro_L-label.nii.gz",
            scene / "femur_right.nii.gz-Necro_R-label.nii.gz",
        ]
        if not ct_path.exists() or any(not p.exists() for p in label_paths):
            continue
        # Healthy
        subject = build_subject(ct_path, label_paths)
        pipeline = CTToXRTransform(necrosis_spots=0)
        healthy = pipeline(subject)["drr"].detach().cpu().clamp(0, 1)
        healthy_name = f"{index:05d}.tiff"
        save_tiff(healthy, output_root / healthy_name)
        metadata[f"{index:05d}"] = 0
        index += 1
        written_since_flush += 1
        # Necrosis with random count and intensity
        necro_spots = random.randint(1, 100)  # noqa: S311
        intensity_pct = random.randint(1, 100)  # noqa: S311
        drop = min(0.95, max(0.05, intensity_pct / 100.0))
        subject_necro = build_subject(ct_path, label_paths)
        pipeline_necro = CTToXRTransform(
            necrosis_spots=necro_spots,
            necrosis_drop=(drop, drop),
        )
        necro = pipeline_necro(subject_necro)["drr"].detach().cpu().clamp(0, 1)
        necro_name = f"{index:05d}.tiff"
        save_tiff(necro, output_root / necro_name)
        metadata[f"{index:05d}"] = necro_spots
        index += 1
        written_since_flush += 1
        if written_since_flush >= flush_every:
            _persist_metadata(output_root, metadata)
            written_since_flush = 0
    if metadata:
        _persist_metadata(output_root, metadata)


def _persist_metadata(output_root: Path, metadata: dict[str, int]) -> None:
    """Safely write metadata.json with atomic replace."""
    meta_path = output_root / "metadata.json"
    tmp_path = meta_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp_path.replace(meta_path)


if __name__ == "__main__":
    main()
