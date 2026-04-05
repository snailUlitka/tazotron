from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest
import torch
import torchio as tio
from diffdrr.data import transform_hu_to_density

from tazotron.datasets.ct import CTDataset
from tazotron.datasets.transforms.necro import AddLateAVNLikeNecrosisV1
from tazotron.datasets.transforms.xray import RenderDRR

if TYPE_CHECKING:
    from pathlib import Path


class _DummyDRR:
    def __init__(self, subject: tio.Subject, **_: object) -> None:
        self.subject = subject
        self.device: torch.device | None = None

    def to(self, device: torch.device | str) -> _DummyDRR:
        self.device = torch.device(device)
        return self

    def __call__(
        self,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        del translations
        volume = self.subject["volume"].data.to(rotations.device)
        projection = volume[0].sum(dim=0, keepdim=True).unsqueeze(0)
        return projection


def _save_image(path: Path, tensor: torch.Tensor, *, label: bool = False) -> None:
    image_cls = tio.LabelMap if label else tio.ScalarImage
    image_cls(tensor=tensor, affine=torch.eye(4)).save(path)


def _make_gradient_ct(shape: tuple[int, int, int], *, base: float) -> torch.Tensor:
    axis_0 = torch.arange(shape[0], dtype=torch.float32)
    axis_1 = torch.arange(shape[1], dtype=torch.float32)
    axis_2 = torch.arange(shape[2], dtype=torch.float32)
    grid_0, grid_1, grid_2 = torch.meshgrid(axis_0, axis_1, axis_2, indexing="ij")
    ct = base + 8.0 * grid_0 + 2.5 * grid_1 + 1.5 * grid_2
    return ct.unsqueeze(0)


@pytest.mark.slow
def test_ct_to_necro_to_xray_pipeline_saves_expected_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tazotron.datasets.transforms.xray.DRR", _DummyDRR)

    patient_dir = tmp_path / "patient-001"
    patient_dir.mkdir(parents=True, exist_ok=True)

    ct_tensor = _make_gradient_ct((16, 16, 16), base=100.0)
    left_label = torch.zeros_like(ct_tensor, dtype=torch.int16)
    left_label[0, 3:12, 4:13, 4:13] = 1
    right_label = torch.zeros_like(left_label)

    ct_path = patient_dir / "ct.nii.gz"
    left_path = patient_dir / "label_femoral_head_left.nii.gz"
    right_path = patient_dir / "label_femoral_head_right.nii.gz"
    femur_left_path = patient_dir / "femur_left.nii.gz.seg.nrrd"
    femur_right_path = patient_dir / "femur_right.nii.gz.seg.nrrd"

    _save_image(ct_path, ct_tensor)
    _save_image(left_path, left_label, label=True)
    _save_image(right_path, right_label, label=True)
    _save_image(femur_left_path, torch.zeros_like(left_label), label=True)
    _save_image(femur_right_path, torch.zeros_like(left_label), label=True)

    necro_config = {
        "probability": 1.0,
        "target_head": "random",
        "severity": "moderate",
        "seed": 42,
        "bone_attenuation_multiplier": 1.0,
    }
    necro = AddLateAVNLikeNecrosisV1(necro_config)
    xray = RenderDRR({"device": "cpu"})
    saved_path = tmp_path / "xray.pt"

    def pipeline(subject: tio.Subject) -> tio.Subject:
        subject["rotations"] = torch.zeros((1, 3), dtype=torch.float32)
        subject["translations"] = torch.zeros((1, 3), dtype=torch.float32)
        subject = necro(subject)
        subject = xray(subject)
        torch.save(subject["xray"].cpu(), saved_path)
        return subject

    dataset = CTDataset(tmp_path, transform=pipeline)
    subject = dataset[0]
    saved_xray = torch.load(saved_path)
    assert torch.equal(saved_xray, subject["xray"])
    assert not torch.equal(subject["volume"].data, ct_tensor)
    expected_density = transform_hu_to_density(subject["volume"].data, 1.0)
    assert torch.allclose(subject["density"].data, expected_density)

    subject_reference = CTDataset(tmp_path)[0]
    subject_reference["rotations"] = torch.zeros((1, 3), dtype=torch.float32)
    subject_reference["translations"] = torch.zeros((1, 3), dtype=torch.float32)
    subject_reference = AddLateAVNLikeNecrosisV1(necro_config)(subject_reference)
    subject_reference = xray(subject_reference)

    assert torch.equal(subject["volume"].data, subject_reference["volume"].data)
    assert torch.equal(subject["xray"], subject_reference["xray"])


@pytest.mark.slow
def test_real_drr_differs_with_necrosis_added(tmp_path: Path) -> None:
    patient_dir = tmp_path / "patient-002"
    patient_dir.mkdir(parents=True, exist_ok=True)

    ct_tensor = _make_gradient_ct((16, 16, 16), base=120.0)
    left_label = torch.zeros_like(ct_tensor, dtype=torch.int16)
    left_label[0, 4:12, 4:12, 4:12] = 1
    right_label = torch.zeros_like(left_label)

    ct_path = patient_dir / "ct.nii.gz"
    left_path = patient_dir / "label_femoral_head_left.nii.gz"
    right_path = patient_dir / "label_femoral_head_right.nii.gz"
    femur_left_path = patient_dir / "femur_left.nii.gz.seg.nrrd"
    femur_right_path = patient_dir / "femur_right.nii.gz.seg.nrrd"

    _save_image(ct_path, ct_tensor)
    _save_image(left_path, left_label, label=True)
    _save_image(right_path, right_label, label=True)
    _save_image(femur_left_path, torch.zeros_like(left_label), label=True)
    _save_image(femur_right_path, torch.zeros_like(left_label), label=True)

    dataset = CTDataset(tmp_path)
    base_subject = dataset[0]
    base_subject["rotations"] = torch.zeros((1, 3), dtype=torch.float32)
    base_subject["translations"] = torch.tensor([[0.0, 800.0, 0.0]], dtype=torch.float32)

    render = RenderDRR({"device": "cpu", "height": 32, "delx": 1.0})

    subject_clean = copy.deepcopy(base_subject)
    clean = render(subject_clean)
    xray_clean = torch.nan_to_num(clean["xray"].detach().cpu())

    subject_necro = copy.deepcopy(base_subject)
    necro = AddLateAVNLikeNecrosisV1(
        {
            "probability": 1.0,
            "target_head": "random",
            "severity": "severe",
            "seed": 42,
            "bone_attenuation_multiplier": 1.0,
        },
    )
    subject_necro = necro(subject_necro)
    necro_rendered = render(subject_necro)
    xray_necro = torch.nan_to_num(necro_rendered["xray"].detach().cpu())

    diff = (xray_necro - xray_clean).abs().sum().item()
    assert diff > 0.0
