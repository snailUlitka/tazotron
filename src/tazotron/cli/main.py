"""CLI entrypoint for tazotron utilities."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from tazotron.datasets.ct import CTDataset
from tazotron.datasets.transforms.crop import BilateralHipROICrop
from tazotron.datasets.transforms.femoral_head import AddFemoralHeadMasks
from tazotron.datasets.transforms.necro import AddRandomNecrosis
from tazotron.datasets.transforms.xray import RenderDRR
from tazotron.datasets.xray import XrayDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

BEST_FEMORAL_HEAD_MASK_PARAMS: dict[str, float | int] = {
    "min_slices_from_top": 5,
    "area_ratio_threshold": 0.70,
    "roundness_ratio_threshold": 1.7,
    "roundness_confirm_slices": 3,
    "roundness_backstep_slices": 0,
}


def _run_xray_dataset_from_ct(data_path: Path, output_path_dir: Path) -> None:
    dataset = CTDataset(data_path)

    with_necro_dir = output_path_dir / "with_necro"
    without_necro_dir = output_path_dir / "without_necro"
    with_necro_dir.mkdir(parents=True, exist_ok=True)
    without_necro_dir.mkdir(parents=True, exist_ok=True)

    crop = BilateralHipROICrop(label_name="label_combined_femoral_head")
    render = RenderDRR({"device": "cpu"})
    necro = AddRandomNecrosis(intensity=0.5, seed=42)

    for index, ct_path in enumerate(
        tqdm(dataset.paths, desc="Rendering XRays", mininterval=2.0),
    ):
        case_name = ct_path.parent.name
        output_with_necro = with_necro_dir / f"{case_name}.pt"
        output_without_necro = without_necro_dir / f"{case_name}.pt"

        if output_with_necro.exists() and output_without_necro.exists():
            continue

        subject = dataset[index]
        subject["rotations"] = torch.zeros((1, 3), dtype=torch.float32)
        subject["translations"] = torch.tensor([[0.0, 800.0, 0.0]], dtype=torch.float32)

        for key in ("volume", "density"):
            if key in subject:
                subject[key].set_data(subject[key].data.to(torch.float32))

        subject = crop(subject)

        with torch.no_grad():
            if not output_without_necro.exists():
                subject_clean = copy.deepcopy(subject)
                clean = render(subject_clean)
                xray_clean = torch.nan_to_num(clean["xray"].detach().cpu())
                XrayDataset.save_pt(xray_clean, output_without_necro)

            if not output_with_necro.exists():
                subject_necro = copy.deepcopy(subject)
                subject_necro = necro(subject_necro)
                necro_rendered = render(subject_necro)
                xray_necro = torch.nan_to_num(necro_rendered["xray"].detach().cpu())
                XrayDataset.save_pt(xray_necro, output_with_necro)


def _run_add_femoral_head_masks(data_path: Path) -> None:
    dataset = CTDataset(data_path)
    add_masks = AddFemoralHeadMasks(overwrite=True, **BEST_FEMORAL_HEAD_MASK_PARAMS)

    for index, ct_path in enumerate(
        tqdm(dataset.paths, desc="Adding femoral head masks", mininterval=2.0),
    ):
        subject = dataset[index]
        subject = add_masks(subject)
        CTDataset.save_femoral_head_masks(subject, ct_path.parent)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tazotron")
    subparsers = parser.add_subparsers(dest="command", required=True)

    xray_parser = subparsers.add_parser(
        "xray_dataset_from_ct",
        help="Render X-rays with and without necrosis from CT volumes.",
    )
    xray_parser.add_argument("data_path", type=Path)
    xray_parser.add_argument("output_path_dir", type=Path)

    add_masks_parser = subparsers.add_parser(
        "add_femoral_head_masks",
        help="Add femoral head masks to CT folders and save them alongside the volume.",
    )
    add_masks_parser.add_argument("data_path", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run CLI commands for dataset processing utilities."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "xray_dataset_from_ct":
        _run_xray_dataset_from_ct(args.data_path, args.output_path_dir)
        return
    if args.command == "add_femoral_head_masks":
        _run_add_femoral_head_masks(args.data_path)
        return

    parser.error(f"Unknown command: {args.command}")
