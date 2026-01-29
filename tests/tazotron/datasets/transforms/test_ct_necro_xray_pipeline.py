from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest
import torch
import torchio as tio

from diffdrr.data import transform_hu_to_density
from tazotron.datasets.ct import CTDataset
from tazotron.datasets.transforms.necro import NECROSIS_HU, AddRandomNecrosis
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


def _expected_ct_after_necrosis(
    ct: torch.Tensor,
    label: torch.Tensor,
    *,
    intensity: float,
    seed: int,
) -> torch.Tensor:
    expected = ct.clone()
    mask = (label[0] == 1) | (label[0] == 2)
    coords = mask.nonzero(as_tuple=False)
    num_necrosis_voxels = int(coords.shape[0] * intensity)
    if num_necrosis_voxels <= 0:
        return expected
    generator = torch.Generator(device=mask.device.type)
    generator.manual_seed(seed)
    perm = torch.randperm(coords.shape[0], generator=generator)[:num_necrosis_voxels]
    selected = coords[perm]
    z, y, x = selected.unbind(dim=1)
    expected[0, z, y, x] = NECROSIS_HU
    return expected


@pytest.mark.slow
def test_ct_to_necro_to_xray_pipeline_saves_expected_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tazotron.datasets.transforms.xray.DRR", _DummyDRR)

    patient_dir = tmp_path / "patient-001"
    patient_dir.mkdir(parents=True, exist_ok=True)

    ct_tensor = torch.full((1, 2, 2, 2), fill_value=100.0, dtype=torch.float32)
    left_label = torch.tensor(
        [[[[1, 0], [0, 1]], [[0, 0], [1, 0]]]],
        dtype=torch.int16,
    )
    right_label = torch.zeros_like(left_label)

    ct_path = patient_dir / "ct.nii.gz"
    left_path = patient_dir / "label_femoral_head_left.nii.gz"
    right_path = patient_dir / "label_femoral_head_right.nii.gz"

    _save_image(ct_path, ct_tensor)
    _save_image(left_path, left_label, label=True)
    _save_image(right_path, right_label, label=True)

    intensity = 0.5
    seed = 42
    necro = AddRandomNecrosis(intensity=intensity, seed=seed)
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

    combined_label = subject["label"].data
    expected_ct = _expected_ct_after_necrosis(
        ct_tensor,
        combined_label,
        intensity=intensity,
        seed=seed,
    )
    expected_xray = expected_ct[0].sum(dim=0, keepdim=True).unsqueeze(0)
    expected_density = transform_hu_to_density(expected_ct, 1.0)

    saved_xray = torch.load(saved_path)
    assert saved_xray.shape == expected_xray.shape
    assert torch.equal(saved_xray, expected_xray)
    assert torch.equal(subject["xray"], expected_xray)
    assert torch.allclose(subject["density"].data, expected_density)


@pytest.mark.slow
def test_real_drr_differs_with_necrosis_added(tmp_path: Path) -> None:
    patient_dir = tmp_path / "patient-002"
    patient_dir.mkdir(parents=True, exist_ok=True)

    ct_tensor = torch.full((1, 8, 8, 8), fill_value=100.0, dtype=torch.float32)
    left_label = torch.zeros_like(ct_tensor, dtype=torch.int16)
    left_label[0, 2:6, 2:6, 2:6] = 1
    right_label = torch.zeros_like(left_label)

    ct_path = patient_dir / "ct.nii.gz"
    left_path = patient_dir / "label_femoral_head_left.nii.gz"
    right_path = patient_dir / "label_femoral_head_right.nii.gz"

    _save_image(ct_path, ct_tensor)
    _save_image(left_path, left_label, label=True)
    _save_image(right_path, right_label, label=True)

    dataset = CTDataset(tmp_path)
    base_subject = dataset[0]
    base_subject["rotations"] = torch.zeros((1, 3), dtype=torch.float32)
    base_subject["translations"] = torch.tensor([[0.0, 800.0, 0.0]], dtype=torch.float32)

    render = RenderDRR({"device": "cpu", "height": 32, "delx": 1.0})

    subject_clean = copy.deepcopy(base_subject)
    clean = render(subject_clean)
    xray_clean = torch.nan_to_num(clean["xray"].detach().cpu())

    subject_necro = copy.deepcopy(base_subject)
    necro = AddRandomNecrosis(intensity=1.0, seed=42)
    subject_necro = necro(subject_necro)
    necro_rendered = render(subject_necro)
    xray_necro = torch.nan_to_num(necro_rendered["xray"].detach().cpu())

    diff = (xray_necro - xray_clean).abs().sum().item()
    assert diff > 0.0
