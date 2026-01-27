from __future__ import annotations

import pytest
import torch
import torchio as tio

from tazotron.datasets.transforms.necro import NECROSIS_HU, AddRandomNecrosis

CT_BASE_HU = 100.0


def _make_subject(ct: torch.Tensor, label: torch.Tensor) -> tio.Subject:
    ct_image = tio.ScalarImage(tensor=ct.clone(), affine=torch.eye(4))
    label_image = tio.LabelMap(tensor=label.clone(), affine=torch.eye(4))
    return tio.Subject({"ct": ct_image, "label": label_image})


class TestAddRandomNecrosis:
    @pytest.mark.fast
    def test_rejects_intensity_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="intensity must be in the range"):
            AddRandomNecrosis(intensity=-0.1)
        with pytest.raises(ValueError, match="intensity must be in the range"):
            AddRandomNecrosis(intensity=1.1)

    @pytest.mark.fast
    def test_is_inplace_on_subject(self) -> None:
        ct = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
        label = torch.ones((1, 2, 2, 2), dtype=torch.int16)
        subject = _make_subject(ct, label)
        transform = AddRandomNecrosis(intensity=0.5, seed=42)

        returned = transform(subject)

        assert returned is subject

    @pytest.mark.fast
    def test_intensity_zero_leaves_ct_unchanged(self) -> None:
        ct = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
        label = torch.ones((1, 2, 2, 2), dtype=torch.int16)
        subject = _make_subject(ct, label)
        transform = AddRandomNecrosis(intensity=0.0, seed=42)

        transform(subject)

        assert torch.equal(subject["ct"].data, ct)

    @pytest.mark.fast
    def test_empty_mask_leaves_ct_unchanged(self) -> None:
        ct = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
        label = torch.zeros((1, 2, 2, 2), dtype=torch.int16)
        subject = _make_subject(ct, label)
        transform = AddRandomNecrosis(intensity=1.0, seed=42)

        transform(subject)

        assert torch.equal(subject["ct"].data, ct)

    @pytest.mark.fast
    def test_intensity_one_sets_all_mask_voxels_to_necrosis_hu(self) -> None:
        ct = torch.full((1, 2, 2, 2), fill_value=CT_BASE_HU, dtype=torch.float32)
        label = torch.tensor(
            [[[[1, 0], [0, 1]], [[1, 0], [0, 0]]]],
            dtype=torch.int16,
        )
        subject = _make_subject(ct, label)
        transform = AddRandomNecrosis(intensity=1.0, seed=42)

        transform(subject)

        mask = subject["label"].data == 1
        masked_values = subject["ct"].data[mask]
        unmasked_values = subject["ct"].data[~mask]
        assert masked_values.numel() > 0
        assert torch.all(masked_values == NECROSIS_HU)
        assert torch.all(unmasked_values == CT_BASE_HU)

    @pytest.mark.fast
    def test_changes_expected_number_of_mask_voxels_for_fractional_intensity(self) -> None:
        ct = torch.full((1, 3, 3, 3), fill_value=200.0, dtype=torch.float32)
        label = torch.ones((1, 3, 3, 3), dtype=torch.int16)
        subject = _make_subject(ct, label)
        transform = AddRandomNecrosis(intensity=0.5, seed=42)

        transform(subject)

        mask = subject["label"].data == 1
        necrosis_mask = (subject["ct"].data == NECROSIS_HU) & mask
        expected = int(mask.sum().item() * 0.5)
        assert int(necrosis_mask.sum().item()) == expected

    @pytest.mark.fast
    def test_seed_makes_selection_deterministic(self) -> None:
        ct = torch.full((1, 3, 3, 3), fill_value=175.0, dtype=torch.float32)
        label = torch.ones((1, 3, 3, 3), dtype=torch.int16)
        subject_a = _make_subject(ct, label)
        subject_b = _make_subject(ct, label)
        transform = AddRandomNecrosis(intensity=0.4, seed=42)

        transform(subject_a)
        transform(subject_b)

        assert torch.equal(subject_a["ct"].data, subject_b["ct"].data)
