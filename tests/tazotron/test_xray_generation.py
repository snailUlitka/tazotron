from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torchio as tio

from tazotron.xray_generation import (
    AUTOPOSE_MODE,
    SKIP_REASON_INVALID_BILATERAL_MASK,
    make_xray_diff_heatmap,
    render_xray_dataset_from_ct,
    xray_to_uint8_image,
)

if TYPE_CHECKING:
    from pathlib import Path

PROJECTION_EPS = 1e-6


class _DummyDRR:
    def __init__(self, subject: tio.Subject, **_: object) -> None:
        self.subject = subject

    def to(self, device: torch.device | str) -> _DummyDRR:
        del device
        return self

    def __call__(self, rotations: torch.Tensor, translations: torch.Tensor, **_: object) -> torch.Tensor:
        del rotations, translations
        volume = self.subject["volume"].data
        projection = volume[0].sum(dim=0, keepdim=True).unsqueeze(0)
        if float(projection.max().item() - projection.min().item()) < PROJECTION_EPS:
            projection = projection.clone()
            projection[..., 0, 0] = 1.0
        return projection


def _save_image(path: Path, tensor: torch.Tensor, *, label: bool = False) -> None:
    image_cls = tio.LabelMap if label else tio.ScalarImage
    image_cls(tensor=tensor, affine=torch.eye(4)).save(path)


def _make_case(root: Path, case_id: str, *, with_right_mask: bool = True) -> None:
    patient_dir = root / case_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    ct_tensor = torch.full((1, 8, 8, 8), fill_value=100.0, dtype=torch.float32)
    left_label = torch.zeros_like(ct_tensor, dtype=torch.int16)
    right_label = torch.zeros_like(ct_tensor, dtype=torch.int16)
    left_label[0, 2:5, 2:5, 2:5] = 1
    if with_right_mask:
        right_label[0, 4:7, 2:5, 4:7] = 1
    _save_image(patient_dir / "ct.nii.gz", ct_tensor)
    _save_image(patient_dir / "label_femoral_head_left.nii.gz", left_label, label=True)
    _save_image(patient_dir / "label_femoral_head_right.nii.gz", right_label, label=True)
    _save_image(patient_dir / "femur_left.nii.gz.seg.nrrd", torch.zeros_like(ct_tensor), label=True)
    _save_image(patient_dir / "femur_right.nii.gz.seg.nrrd", torch.zeros_like(ct_tensor), label=True)


@pytest.mark.slow
def test_render_xray_dataset_from_ct_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("tazotron.datasets.transforms.xray.DRR", _DummyDRR)
    _make_case(tmp_path, "s0001")
    output_root = tmp_path / "output"

    render_xray_dataset_from_ct(tmp_path, output_root, framing_mode=AUTOPOSE_MODE, device="cpu")

    assert (output_root / "with_necro" / "s0001.pt").is_file()
    assert (output_root / "without_necro" / "s0001.pt").is_file()
    assert not (output_root / "skipped_xray_cases.tsv").exists()


@pytest.mark.slow
def test_render_xray_dataset_from_ct_logs_invalid_mask_skip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("tazotron.datasets.transforms.xray.DRR", _DummyDRR)
    _make_case(tmp_path, "s0002", with_right_mask=False)
    output_root = tmp_path / "output"

    render_xray_dataset_from_ct(tmp_path, output_root, framing_mode=AUTOPOSE_MODE, device="cpu")

    skipped_path = output_root / "skipped_xray_cases.tsv"
    assert skipped_path.is_file()
    assert SKIP_REASON_INVALID_BILATERAL_MASK in skipped_path.read_text(encoding="utf-8")


def test_xray_to_uint8_image_normalizes_to_full_range() -> None:
    tensor = torch.tensor([[0.0, 2.0], [4.0, 8.0]], dtype=torch.float32)
    image = xray_to_uint8_image(tensor)
    assert image.dtype == torch.uint8
    assert int(image.min().item()) == 0
    assert int(image.max().item()) == 255


def test_make_xray_diff_heatmap_marks_brightening_in_red() -> None:
    before = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    after = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    heatmap = make_xray_diff_heatmap(before, after, sensitivity=12.0)

    assert heatmap.shape == (2, 2, 3)
    assert heatmap.dtype == torch.uint8
    assert tuple(heatmap[0, 0].tolist()) == (0, 0, 255)
    assert heatmap[1, 1, 0].item() == 255
    assert heatmap[1, 1, 2].item() == 0
