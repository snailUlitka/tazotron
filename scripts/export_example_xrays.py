"""Export before/after necrosis X-ray examples and a heat-diff image for one CT case."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from tazotron.datasets.ct import CTDataset, FEMORAL_HEAD_LEFT, FEMORAL_HEAD_RIGHT, FEMUR_LEFT, FEMUR_RIGHT
from tazotron.datasets.transforms.femoral_head import AddFemoralHeadMasks
from tazotron.datasets.transforms.necro import AddLateAVNLikeNecrosisV1
from tazotron.datasets.transforms.xray import RenderDRR
from tazotron.xray_generation import (
    _cast_volume_to_float32,
    apply_framing,
    make_xray_diff_heatmap,
    save_uint8_image,
    xray_to_uint8_image,
)

BEST_FEMORAL_HEAD_MASK_PARAMS: dict[str, float | int] = {
    "min_slices_from_top": 5,
    "area_ratio_threshold": 0.70,
    "roundness_ratio_threshold": 1.7,
    "roundness_confirm_slices": 3,
    "roundness_backstep_slices": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render X-ray example images for one CT case using autopose and the new necrosis generator.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Path to a case directory containing ct.nii.gz and segmentation files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: .data/examples/<case-id>).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for DRR rendering (default: cpu).",
    )
    parser.add_argument(
        "--save-tiff",
        action="store_true",
        help="Also save TIFF copies alongside PNG files.",
    )
    return parser.parse_args()


def load_case_subject(case_dir: Path):
    dataset = CTDataset(case_dir.parent)
    expected_ct_path = (case_dir / "ct.nii.gz").resolve()
    for index, ct_path in enumerate(dataset.paths):
        if ct_path.resolve() == expected_ct_path:
            subject = dataset[index]
            return subject
    msg = f"Case {case_dir} is not discoverable by CTDataset rooted at {case_dir.parent}"
    raise ValueError(msg)


def ensure_femoral_head_masks(subject):
    if subject.get(FEMORAL_HEAD_LEFT) is not None and subject.get(FEMORAL_HEAD_RIGHT) is not None:
        return subject
    if subject.get(FEMUR_LEFT) is None or subject.get(FEMUR_RIGHT) is None:
        msg = "Case must contain femoral head masks or femur segmentations to derive them."
        raise ValueError(msg)
    add_masks = AddFemoralHeadMasks(overwrite=True, **BEST_FEMORAL_HEAD_MASK_PARAMS)
    return add_masks(subject)


def save_example_images(
    *,
    clean_xray,
    necro_xray,
    output_dir: Path,
    save_tiff: bool,
) -> None:
    clean_image = xray_to_uint8_image(clean_xray)
    necro_image = xray_to_uint8_image(necro_xray)
    diff_image = make_xray_diff_heatmap(clean_xray, necro_xray)

    save_uint8_image(clean_image, output_dir / "xray_before_necrosis.png")
    save_uint8_image(necro_image, output_dir / "xray_after_necrosis.png")
    save_uint8_image(diff_image, output_dir / "xray_heat_diff.png")

    if save_tiff:
        save_uint8_image(clean_image, output_dir / "xray_before_necrosis.tiff")
        save_uint8_image(necro_image, output_dir / "xray_after_necrosis.tiff")
        save_uint8_image(diff_image, output_dir / "xray_heat_diff.tiff")


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    if not case_dir.is_dir():
        msg = f"Case directory does not exist: {case_dir}"
        raise FileNotFoundError(msg)

    output_dir = args.output_dir.resolve() if args.output_dir is not None else Path(".data/examples") / case_dir.name
    subject = ensure_femoral_head_masks(load_case_subject(case_dir))
    _cast_volume_to_float32(subject)
    framed = apply_framing(subject, framing_mode="autopose")

    render = RenderDRR({"device": args.device})
    necro = AddLateAVNLikeNecrosisV1(
        {
            "probability": 1.0,
            "target_head": "random",
            "severity": "moderate",
            "seed": 42,
            "bone_attenuation_multiplier": 1.0,
        },
    )

    with torch.no_grad():
        clean = render(copy.deepcopy(framed))
        subject_necro = necro(copy.deepcopy(framed))
        necro_rendered = render(subject_necro)

    save_example_images(
        clean_xray=clean["xray"],
        necro_xray=necro_rendered["xray"],
        output_dir=output_dir,
        save_tiff=args.save_tiff,
    )

    print(f"Saved example images to {output_dir}")


if __name__ == "__main__":
    main()
