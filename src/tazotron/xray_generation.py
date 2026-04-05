"""Helpers for rendering synthetic X-ray datasets from CT volumes."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import torch
from diffdrr.drr import DRR
from tqdm import tqdm

from tazotron.datasets.ct import COMBINED_FEMORAL_HEAD, CTDataset
from tazotron.datasets.transforms.crop import BilateralHipROICrop
from tazotron.datasets.transforms.necro import AddLateAVNLikeNecrosisV1
from tazotron.datasets.transforms.pose import AutoBilateralHipPose, InvalidBilateralMaskError
from tazotron.datasets.transforms.xray import EMPTY_EPS, RenderDRR
from tazotron.datasets.xray import XrayDataset

if TYPE_CHECKING:
    from pathlib import Path

    import torchio as tio

LEGACY_CROP_MODE = "legacy_crop"
AUTOPOSE_MODE = "autopose"
FRAMING_MODES = (LEGACY_CROP_MODE, AUTOPOSE_MODE)
SKIP_REASON_INVALID_BILATERAL_MASK = "invalid_bilateral_mask"
SKIP_REASON_EMPTY_DRR = "empty_drr"
XRAY_BATCH_RANK = 4
XRAY_CHANNEL_RANK = 3
XRAY_IMAGE_RANK = 2

FramingMode = Literal["legacy_crop", "autopose"]


@dataclass(frozen=True)
class RenderBenchmarkRow:
    """Per-case render/framing metrics."""

    case_id: str
    framing_mode: str
    status: str
    reason: str
    sid_mm: float | None
    sdd_mm: float | None
    delx_mm: float | None
    dely_mm: float | None
    width_px: int | None
    height_px: int | None
    projected_bbox_width_px: int | None
    projected_bbox_height_px: int | None
    bbox_center_offset_x_px: float | None
    bbox_center_offset_y_px: float | None
    min_edge_margin_px: float | None
    projected_area_ratio: float | None
    border_touch: int | None


def apply_framing(subject: tio.Subject, *, framing_mode: FramingMode) -> tio.Subject:
    """Attach camera pose/FOV information for the requested framing mode."""
    if framing_mode == LEGACY_CROP_MODE:
        crop = BilateralHipROICrop(label_name=COMBINED_FEMORAL_HEAD)
        subject["rotations"] = torch.zeros((1, 3), dtype=torch.float32)
        subject["translations"] = torch.tensor([[0.0, 800.0, 0.0]], dtype=torch.float32)
        subject["drr_config"] = {
            "sdd": 1020.0,
            "width": 200,
            "height": 200,
            "delx": 2.0,
            "dely": 2.0,
            "reverse_x_axis": True,
        }
        subject["pose_metadata"] = {
            "mode": LEGACY_CROP_MODE,
            "sid_mm": 800.0,
            "sdd_mm": 1020.0,
        }
        return crop(subject)
    if framing_mode == AUTOPOSE_MODE:
        return AutoBilateralHipPose()(subject)

    msg = f"Unknown framing mode: {framing_mode}"
    raise ValueError(msg)


def render_xray_dataset_from_ct(
    data_path: Path,
    output_path_dir: Path,
    *,
    framing_mode: FramingMode = AUTOPOSE_MODE,
    device: str = "cpu",
    overwrite_existing: bool = False,
) -> None:
    """Render X-rays with and without necrosis from CT volumes."""
    dataset = CTDataset(data_path)
    with_necro_dir = output_path_dir / "with_necro"
    without_necro_dir = output_path_dir / "without_necro"
    with_necro_dir.mkdir(parents=True, exist_ok=True)
    without_necro_dir.mkdir(parents=True, exist_ok=True)

    render = RenderDRR({"device": device})
    necro = AddLateAVNLikeNecrosisV1(
        {
            "probability": 1.0,
            "target_head": "random",
            "severity": "moderate",
            "seed": 42,
            "bone_attenuation_multiplier": 1.0,
        },
    )
    skipped_rows: list[tuple[str, str]] = []

    for index, ct_path in enumerate(tqdm(dataset.paths, desc="Rendering XRays", mininterval=2.0)):
        case_id = ct_path.parent.name
        output_with_necro = with_necro_dir / f"{case_id}.pt"
        output_without_necro = without_necro_dir / f"{case_id}.pt"
        if not overwrite_existing and output_with_necro.exists() and output_without_necro.exists():
            continue

        subject = dataset[index]
        _cast_volume_to_float32(subject)
        try:
            framed = apply_framing(subject, framing_mode=framing_mode)
            with torch.no_grad():
                clean = render(copy.deepcopy(framed))
                subject_necro = necro(copy.deepcopy(framed))
                necro_rendered = render(subject_necro)

            clean_xray = torch.nan_to_num(clean["xray"].detach().cpu())
            necro_xray = torch.nan_to_num(necro_rendered["xray"].detach().cpu())
            if overwrite_existing or not output_without_necro.exists():
                XrayDataset.save_pt(clean_xray, output_without_necro)
            if overwrite_existing or not output_with_necro.exists():
                XrayDataset.save_pt(necro_xray, output_with_necro)
        except InvalidBilateralMaskError:
            output_without_necro.unlink(missing_ok=True)
            output_with_necro.unlink(missing_ok=True)
            skipped_rows.append((case_id, SKIP_REASON_INVALID_BILATERAL_MASK))
            tqdm.write(f"[WARN] Skipping {case_id}: invalid bilateral femoral-head mask")
        except ValueError as exc:
            if "Rendered DRR is empty" not in str(exc):
                raise
            output_without_necro.unlink(missing_ok=True)
            output_with_necro.unlink(missing_ok=True)
            skipped_rows.append((case_id, SKIP_REASON_EMPTY_DRR))
            tqdm.write(f"[WARN] Skipping {case_id}: {exc}")

    _write_skipped_rows(output_path_dir / "skipped_xray_cases.tsv", skipped_rows)


def benchmark_render_mode(
    data_path: Path,
    *,
    framing_mode: FramingMode,
    device: str = "cpu",
) -> list[RenderBenchmarkRow]:
    """Render every case and collect framing-quality metrics."""
    dataset = CTDataset(data_path)
    render = RenderDRR({"device": device})
    rows: list[RenderBenchmarkRow] = []

    for index, ct_path in enumerate(tqdm(dataset.paths, desc=f"Benchmark {framing_mode}", mininterval=2.0)):
        case_id = ct_path.parent.name
        subject = dataset[index]
        _cast_volume_to_float32(subject)
        try:
            framed = apply_framing(subject, framing_mode=framing_mode)
            rendered = render(copy.deepcopy(framed))
            drr_config = render.resolve_config(framed)
            projection = _render_mask_projection(framed, drr_config)
            rows.append(
                RenderBenchmarkRow(
                    case_id=case_id,
                    framing_mode=framing_mode,
                    status="success",
                    reason="",
                    sid_mm=_sid_from_metadata(framed),
                    sdd_mm=float(drr_config.sdd),
                    delx_mm=float(drr_config.delx),
                    dely_mm=float(drr_config.dely if drr_config.dely is not None else drr_config.delx),
                    width_px=int(drr_config.width or drr_config.height),
                    height_px=int(drr_config.height),
                    **compute_projection_bbox_metrics(projection),
                ),
            )
            del rendered
        except InvalidBilateralMaskError:
            rows.append(
                RenderBenchmarkRow(
                    case_id=case_id,
                    framing_mode=framing_mode,
                    status="skipped",
                    reason=SKIP_REASON_INVALID_BILATERAL_MASK,
                    sid_mm=None,
                    sdd_mm=None,
                    delx_mm=None,
                    dely_mm=None,
                    width_px=None,
                    height_px=None,
                    projected_bbox_width_px=None,
                    projected_bbox_height_px=None,
                    bbox_center_offset_x_px=None,
                    bbox_center_offset_y_px=None,
                    min_edge_margin_px=None,
                    projected_area_ratio=None,
                    border_touch=None,
                ),
            )
        except ValueError as exc:
            if "Rendered DRR is empty" not in str(exc):
                raise
            rows.append(
                RenderBenchmarkRow(
                    case_id=case_id,
                    framing_mode=framing_mode,
                    status="skipped",
                    reason=SKIP_REASON_EMPTY_DRR,
                    sid_mm=None,
                    sdd_mm=None,
                    delx_mm=None,
                    dely_mm=None,
                    width_px=None,
                    height_px=None,
                    projected_bbox_width_px=None,
                    projected_bbox_height_px=None,
                    bbox_center_offset_x_px=None,
                    bbox_center_offset_y_px=None,
                    min_edge_margin_px=None,
                    projected_area_ratio=None,
                    border_touch=None,
                ),
            )
    return rows


def compute_projection_bbox_metrics(projection: torch.Tensor) -> dict[str, float | int | None]:
    """Compute 2D framing metrics from a binary-mask projection."""
    image = squeeze_xray_tensor(projection)
    active = image > EMPTY_EPS
    if not active.any():
        return {
            "projected_bbox_width_px": None,
            "projected_bbox_height_px": None,
            "bbox_center_offset_x_px": None,
            "bbox_center_offset_y_px": None,
            "min_edge_margin_px": None,
            "projected_area_ratio": None,
            "border_touch": None,
        }

    points = active.nonzero(as_tuple=False)
    row_min = int(points[:, 0].min().item())
    row_max = int(points[:, 0].max().item())
    col_min = int(points[:, 1].min().item())
    col_max = int(points[:, 1].max().item())
    height, width = image.shape
    bbox_width = col_max - col_min + 1
    bbox_height = row_max - row_min + 1
    center_col = (col_min + col_max) / 2.0
    center_row = (row_min + row_max) / 2.0
    frame_center_col = (width - 1) / 2.0
    frame_center_row = (height - 1) / 2.0
    min_edge_margin = float(min(col_min, width - 1 - col_max, row_min, height - 1 - row_max))
    projected_area_ratio = float(active.sum().item()) / float(height * width)
    return {
        "projected_bbox_width_px": bbox_width,
        "projected_bbox_height_px": bbox_height,
        "bbox_center_offset_x_px": center_col - frame_center_col,
        "bbox_center_offset_y_px": center_row - frame_center_row,
        "min_edge_margin_px": min_edge_margin,
        "projected_area_ratio": projected_area_ratio,
        "border_touch": int(min_edge_margin <= 0.0),
    }


def rows_to_dicts(rows: list[RenderBenchmarkRow]) -> list[dict[str, object]]:
    """Convert dataclass rows to dictionaries for CSV serialization."""
    return [asdict(row) for row in rows]


def squeeze_xray_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Squeeze XR/DRR tensors to 2D image format."""
    if tensor.ndim == XRAY_BATCH_RANK and tensor.shape[:2] == (1, 1):
        return tensor[0, 0].detach().cpu().to(torch.float32)
    if tensor.ndim == XRAY_CHANNEL_RANK and tensor.shape[0] == 1:
        return tensor[0].detach().cpu().to(torch.float32)
    if tensor.ndim == XRAY_IMAGE_RANK:
        return tensor.detach().cpu().to(torch.float32)
    msg = f"Unsupported X-ray tensor shape: {tuple(tensor.shape)}"
    raise ValueError(msg)


def _cast_volume_to_float32(subject: tio.Subject) -> None:
    for key in ("volume", "density"):
        if key in subject:
            subject[key].set_data(subject[key].data.to(torch.float32))


def _render_mask_projection(subject: tio.Subject, config) -> torch.Tensor:
    mask_subject = copy.deepcopy(subject)
    mask = (mask_subject[COMBINED_FEMORAL_HEAD].data > 0).to(torch.float32)
    for key in ("volume", "density"):
        if key in mask_subject:
            mask_subject[key].set_data(mask.clone())

    drr_module = DRR(
        mask_subject,
        sdd=config.sdd,
        height=config.height,
        delx=config.delx,
        width=config.width,
        dely=config.dely,
        x0=config.x0,
        y0=config.y0,
        reverse_x_axis=config.reverse_x_axis,
        patch_size=config.patch_size,
        renderer=config.renderer,
    )
    target_device = config.resolved_device() or subject["rotations"].device
    rotations = subject["rotations"].to(target_device)
    translations = subject["translations"].to(target_device)
    drr_module = drr_module.to(target_device)
    projection = drr_module(rotations, translations, parameterization="euler_angles", convention="ZXY")
    return projection.detach().cpu()


def _sid_from_metadata(subject: tio.Subject) -> float | None:
    metadata = subject.get("pose_metadata")
    if isinstance(metadata, dict):
        sid = metadata.get("sid_mm")
        if sid is not None:
            return float(sid)
    return None


def _write_skipped_rows(path: Path, rows: list[tuple[str, str]]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("case_id\treason\n")
        for case_id, reason in rows:
            handle.write(f"{case_id}\t{reason}\n")
