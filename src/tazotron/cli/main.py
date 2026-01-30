"""CLI entrypoint for tazotron utilities."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Sequence
from pathlib import Path

import torch
from tqdm import tqdm

from tazotron.datasets.ct import CTDataset
from tazotron.datasets.transforms.necro import AddRandomNecrosis
from tazotron.datasets.transforms.xray import RenderDRR
from tazotron.datasets.xray import XrayDataset


def _run_xray_dataset_from_ct(data_path: Path, output_path_dir: Path) -> None:
    dataset = CTDataset(
        data_path,
        ct_name="ct.nii.gz",
        label_femoral_head_left="label_femoral_head_left.nii.gz",
        label_femoral_head_right="label_femoral_head_right.nii.gz",
    )

    with_necro_dir = output_path_dir / "with_necro"
    without_necro_dir = output_path_dir / "without_necro"
    with_necro_dir.mkdir(parents=True, exist_ok=True)
    without_necro_dir.mkdir(parents=True, exist_ok=True)

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tazotron")
    subparsers = parser.add_subparsers(dest="command", required=True)

    xray_parser = subparsers.add_parser(
        "xray_dataset_from_ct",
        help="Render X-rays with and without necrosis from CT volumes.",
    )
    xray_parser.add_argument("data_path", type=Path)
    xray_parser.add_argument("output_path_dir", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "xray_dataset_from_ct":
        _run_xray_dataset_from_ct(args.data_path, args.output_path_dir)
        return

    parser.error(f"Unknown command: {args.command}")
