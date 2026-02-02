import pytest
import torch
import torchio as tio

from tazotron.datasets.transforms.crop import BilateralHipROICrop


def _make_subject(
    *,
    shape: tuple[int, int, int] = (10, 12, 20),
    left: tuple[int, int, int] = (2, 3, 4),
    right: tuple[int, int, int] = (7, 8, 8),
    left_id: int = 1,
    right_id: int = 2,
    affine: torch.Tensor | None = None,
) -> tio.Subject:
    label = torch.zeros((1, *shape), dtype=torch.int64)
    label[0, left[0], left[1], left[2]] = left_id
    label[0, right[0], right[1], right[2]] = right_id
    image = torch.zeros((1, *shape), dtype=torch.float32)
    if affine is None:
        affine = torch.eye(4, dtype=torch.float32)
    label_map = tio.LabelMap(tensor=label, affine=affine)
    intensity = tio.ScalarImage(tensor=image, affine=affine)
    return tio.Subject({"label": label_map, "image": intensity, "meta": "keep"})


@pytest.mark.fast
def test_rejects_non_subject_input() -> None:
    transform = BilateralHipROICrop()
    with pytest.raises(TypeError, match="torchio.Subject"):
        transform({"label": None})


@pytest.mark.fast
def test_requires_both_masks_present() -> None:
    subject = _make_subject()
    subject["label"].data.zero_()
    transform = BilateralHipROICrop()
    with pytest.raises(ValueError, match="Both left and right hip masks must be present"):
        transform(subject)


@pytest.mark.fast
def test_crops_bilateral_roi_with_min_lr_size() -> None:
    subject = _make_subject()
    transform = BilateralHipROICrop(min_lr_mm=10.0, margin_lr_mm=0.0)
    cropped = transform(subject)

    assert isinstance(cropped, tio.Subject)
    assert cropped["meta"] == "keep"

    cropped_label = cropped["label"].data[0]
    cropped_image = cropped["image"].data[0]

    assert cropped_label.shape == cropped_image.shape
    assert cropped_label.shape[-1] == 11
    assert cropped_label.shape[-2] == subject["label"].data.shape[-2]
    assert cropped_label.shape[-3] == 6
    assert (cropped_label == 1).any()
    assert (cropped_label == 2).any()


@pytest.mark.fast
def test_affine_translates_with_crop_offset() -> None:
    subject = _make_subject()
    transform = BilateralHipROICrop(min_lr_mm=10.0, margin_lr_mm=0.0)
    cropped = transform(subject)

    affine = cropped["image"].affine
    expected_offset = torch.tensor([2.0, 0.0, 1.0], dtype=torch.float64)

    assert torch.allclose(torch.from_numpy(affine[:3, 3]), expected_offset)


@pytest.mark.fast
def test_anisotropic_spacing_affects_lr_voxels() -> None:
    affine = torch.diag(torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=torch.float32))
    subject = _make_subject(affine=affine)
    transform = BilateralHipROICrop(min_lr_mm=10.0, margin_lr_mm=0.0)
    cropped = transform(subject)

    spacing_x = subject["label"].spacing[2]
    assert spacing_x == 2.0
    assert cropped["label"].data.shape[-1] == 5


@pytest.mark.fast
def test_clamps_crop_to_volume_bounds() -> None:
    subject = _make_subject(
        shape=(6, 6, 6),
        left=(0, 1, 0),
        right=(5, 4, 1),
    )
    transform = BilateralHipROICrop(min_lr_mm=20.0, margin_lr_mm=0.0)
    cropped = transform(subject)

    assert cropped["label"].data.shape[-1] == 6
    assert cropped["image"].data.shape[-3:] == (6, 6, 6)
    assert torch.allclose(
        torch.from_numpy(cropped["label"].affine[:3, 3]),
        torch.zeros(3, dtype=torch.float64),
    )


@pytest.mark.fast
def test_large_min_lr_mm_expands_to_full_width() -> None:
    subject = _make_subject(shape=(8, 8, 10), right=(7, 7, 8))
    transform = BilateralHipROICrop(min_lr_mm=1_000.0, margin_lr_mm=0.0)
    cropped = transform(subject)

    assert cropped["label"].data.shape[-1] == 10


@pytest.mark.fast
def test_affine_preserves_existing_translation() -> None:
    affine = torch.eye(4, dtype=torch.float32)
    affine[:3, 3] = torch.tensor([10.0, -5.0, 3.0])
    subject = _make_subject(affine=affine)
    transform = BilateralHipROICrop(min_lr_mm=10.0, margin_lr_mm=0.0)
    cropped = transform(subject)

    expected_offset = torch.tensor([12.0, -5.0, 4.0], dtype=torch.float64)
    assert torch.allclose(torch.from_numpy(cropped["image"].affine[:3, 3]), expected_offset)
