"""Export before/after necrosis X-ray examples and a heat-diff image for one CT case."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import torch

from tazotron.datasets.ct import (
    FEMORAL_HEAD_LEFT,
    FEMORAL_HEAD_RIGHT,
    FEMUR_LEFT,
    FEMUR_RIGHT,
    CTDataset,
)
from tazotron.datasets.transforms.femoral_head import AddFemoralHeadMasks
from tazotron.datasets.transforms.necro import (
    LATE_AVN_DATASET_DEFAULT_CONFIG,
    LATE_AVN_REFERENCE_VARIANT_004,
    LATE_AVN_REFERENCE_VARIANT_005,
    AddLateAVNLikeNecrosisV1,
    AddLateAVNLikeNecrosisV1Config,
)
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
DEFAULT_EXAMPLE_NECRO_CONFIG = AddLateAVNLikeNecrosisV1Config.model_validate(
    {
        **LATE_AVN_DATASET_DEFAULT_CONFIG,
        "seed": 42,
    },
)


def parse_float_pair(value: str) -> tuple[float, float]:
    left, right = value.split(",", maxsplit=1)
    return float(left), float(right)


def parse_float_triple(value: str) -> tuple[float, float, float]:
    first, second, third = value.split(",", maxsplit=2)
    return float(first), float(second), float(third)


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
    parser.add_argument(
        "--batch-presets",
        action="store_true",
        help="Render a bundle of necrosis variants for the same case.",
    )
    parser.add_argument(
        "--target-head",
        choices=("left", "right", "both", "random"),
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.target_head,
        help="Which femoral head to edit (default: random).",
    )
    parser.add_argument(
        "--target-head-weights",
        type=parse_float_triple,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.target_head_weights,
        metavar="LEFT,RIGHT,BOTH",
        help="Weights used when --target-head=random.",
    )
    parser.add_argument(
        "--severity",
        choices=("mild", "moderate", "severe", "random"),
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.severity,
        help="Necrosis severity preset or random weighted sampling (default: random).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.seed,
        help="Random seed for the necrosis transform (default: 42).",
    )
    parser.add_argument(
        "--bone-attenuation-multiplier",
        type=float,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.bone_attenuation_multiplier,
        help="Density recomputation multiplier after the edit (default: 1.0).",
    )
    parser.add_argument(
        "--severity-weights",
        type=parse_float_triple,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.severity_weights,
        metavar="MILD,MODERATE,SEVERE",
        help="Weights used when --severity=random.",
    )
    parser.add_argument(
        "--effect-strength-range",
        type=parse_float_pair,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.effect_strength_range,
        metavar="MIN,MAX",
        help="Range for low/high remodeling strength multiplier.",
    )
    parser.add_argument(
        "--collapse-strength-range",
        type=parse_float_pair,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.collapse_strength_range,
        metavar="MIN,MAX",
        help="Range for subchondral collapse strength multiplier.",
    )
    parser.add_argument(
        "--angle-jitter-deg",
        type=float,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.angle_jitter_deg,
        help="Angular jitter applied to the severity preset in degrees.",
    )
    parser.add_argument(
        "--depth-jitter-ratio",
        type=float,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.depth_jitter_ratio,
        help="Relative jitter for sector depth ratio.",
    )
    parser.add_argument(
        "--shell-depth-jitter-ratio",
        type=float,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.shell_depth_jitter_ratio,
        help="Relative jitter for shell depth ratio.",
    )
    parser.add_argument(
        "--blob-count-jitter",
        type=int,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.blob_count_jitter,
        help="Integer jitter applied to preset blob counts.",
    )
    parser.add_argument(
        "--blob-size-jitter-ratio",
        type=float,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.blob_size_jitter_ratio,
        help="Relative jitter for blob sigma sizes.",
    )
    parser.add_argument(
        "--ap-jitter-range",
        type=parse_float_pair,
        default=DEFAULT_EXAMPLE_NECRO_CONFIG.ap_jitter_range,
        metavar="MIN,MAX",
        help="Anterior-posterior pole jitter range.",
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


def build_necro_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "probability": 1.0,
        "target_head": args.target_head,
        "target_head_weights": args.target_head_weights,
        "severity": args.severity,
        "seed": args.seed,
        "bone_attenuation_multiplier": args.bone_attenuation_multiplier,
        "severity_weights": args.severity_weights,
        "effect_strength_range": args.effect_strength_range,
        "collapse_strength_range": args.collapse_strength_range,
        "angle_jitter_deg": args.angle_jitter_deg,
        "depth_jitter_ratio": args.depth_jitter_ratio,
        "shell_depth_jitter_ratio": args.shell_depth_jitter_ratio,
        "blob_count_jitter": args.blob_count_jitter,
        "blob_size_jitter_ratio": args.blob_size_jitter_ratio,
        "ap_jitter_range": args.ap_jitter_range,
    }


def batch_metadata_row(
    *,
    variant_id: str,
    args: argparse.Namespace,
    necro_config: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "variant_id": variant_id,
        "severity": args.severity,
        "target_head_weights": ",".join(str(value) for value in args.target_head_weights),
        "severity_weights": ",".join(str(value) for value in args.severity_weights),
        "effect_strength_range": ",".join(str(value) for value in args.effect_strength_range),
        "collapse_strength_range": ",".join(str(value) for value in args.collapse_strength_range),
        "angle_jitter_deg": args.angle_jitter_deg,
        "depth_jitter_ratio": args.depth_jitter_ratio,
        "shell_depth_jitter_ratio": args.shell_depth_jitter_ratio,
        "blob_size_jitter_ratio": args.blob_size_jitter_ratio,
        "blob_count_jitter": args.blob_count_jitter,
        "ap_jitter_range": ",".join(str(value) for value in args.ap_jitter_range),
        "target_head": args.target_head,
        "seed": args.seed,
        "bone_attenuation_multiplier": args.bone_attenuation_multiplier,
        "before_png": f"{variant_id}__before.png",
        "after_png": f"{variant_id}__after.png",
        "diff_heat_png": f"{variant_id}__diff_heat.png",
    }
    del necro_config
    return row


def save_variant_images(
    *,
    clean_xray,
    necro_xray,
    output_dir: Path,
    variant_id: str,
    save_tiff: bool,
) -> None:
    clean_image = xray_to_uint8_image(clean_xray)
    necro_image = xray_to_uint8_image(necro_xray)
    diff_image = make_xray_diff_heatmap(clean_xray, necro_xray)

    save_uint8_image(clean_image, output_dir / f"{variant_id}__before.png")
    save_uint8_image(necro_image, output_dir / f"{variant_id}__after.png")
    save_uint8_image(diff_image, output_dir / f"{variant_id}__diff_heat.png")

    if save_tiff:
        save_uint8_image(clean_image, output_dir / f"{variant_id}__before.tiff")
        save_uint8_image(necro_image, output_dir / f"{variant_id}__after.tiff")
        save_uint8_image(diff_image, output_dir / f"{variant_id}__diff_heat.tiff")


def batch_variants(args: argparse.Namespace) -> list[argparse.Namespace]:
    def variant(**overrides: object) -> argparse.Namespace:
        payload = vars(args).copy()
        payload.update(overrides)
        return argparse.Namespace(**payload)

    return [
        variant(
            **LATE_AVN_DATASET_DEFAULT_CONFIG,
        ),
        variant(
            severity="mild",
            severity_weights=(0.7, 0.2, 0.1),
            effect_strength_range=(0.95, 1.08),
            collapse_strength_range=(1.00, 1.08),
            angle_jitter_deg=3.0,
            depth_jitter_ratio=0.10,
            shell_depth_jitter_ratio=0.10,
            blob_size_jitter_ratio=0.10,
            blob_count_jitter=0,
            ap_jitter_range=(-0.12, 0.12),
        ),
        variant(
            severity="moderate",
            severity_weights=(0.2, 0.7, 0.1),
            effect_strength_range=(1.08, 1.24),
            collapse_strength_range=(1.02, 1.12),
            angle_jitter_deg=6.0,
            depth_jitter_ratio=0.18,
            shell_depth_jitter_ratio=0.10,
            blob_size_jitter_ratio=0.22,
            blob_count_jitter=2,
            ap_jitter_range=(-0.18, 0.18),
        ),
        variant(
            **LATE_AVN_REFERENCE_VARIANT_004,
        ),
        variant(
            **LATE_AVN_REFERENCE_VARIANT_005,
        ),
    ]


def render_variant(framed, *, device: str, necro_config: dict[str, object]):
    render = RenderDRR({"device": device})
    necro = AddLateAVNLikeNecrosisV1(necro_config)
    with torch.no_grad():
        clean = render(copy.deepcopy(framed))
        subject_necro = necro(copy.deepcopy(framed))
        necro_rendered = render(subject_necro)
    return clean["xray"], necro_rendered["xray"]


def write_batch_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant_id",
        "severity",
        "target_head_weights",
        "severity_weights",
        "effect_strength_range",
        "collapse_strength_range",
        "angle_jitter_deg",
        "depth_jitter_ratio",
        "shell_depth_jitter_ratio",
        "blob_size_jitter_ratio",
        "blob_count_jitter",
        "ap_jitter_range",
        "target_head",
        "seed",
        "bone_attenuation_multiplier",
        "before_png",
        "after_png",
        "diff_heat_png",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    if args.batch_presets:
        batch_output_dir = output_dir / "batch"
        metadata_rows: list[dict[str, object]] = []
        for index, variant_args in enumerate(batch_variants(args), start=1):
            necro_config = build_necro_config(variant_args)
            clean_xray, necro_xray = render_variant(framed, device=args.device, necro_config=necro_config)
            variant_id = f"variant_{index:03d}"
            save_variant_images(
                clean_xray=clean_xray,
                necro_xray=necro_xray,
                output_dir=batch_output_dir,
                variant_id=variant_id,
                save_tiff=args.save_tiff,
            )
            metadata_rows.append(
                batch_metadata_row(
                    variant_id=variant_id,
                    args=variant_args,
                    necro_config=necro_config,
                ),
            )
            print(f"Saved batch variant to {batch_output_dir}: {variant_id}")
            print(f"Necrosis config: {necro_config}")
        metadata_path = batch_output_dir / "metadata.csv"
        write_batch_metadata(metadata_path, metadata_rows)
        print(f"Saved batch metadata to {metadata_path}")
        return

    clean_xray, necro_xray = render_variant(framed, device=args.device, necro_config=build_necro_config(args))
    save_example_images(
        clean_xray=clean_xray,
        necro_xray=necro_xray,
        output_dir=output_dir,
        save_tiff=args.save_tiff,
    )

    print(f"Saved example images to {output_dir}")
    print(f"Necrosis config: {build_necro_config(args)}")


if __name__ == "__main__":
    main()
