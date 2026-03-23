import pytest
import torch
import torchio as tio

from tazotron.datasets.transforms.pose import AutoBilateralHipPose, InvalidBilateralMaskError

RIGHT_LABEL_ID = 2
EXPECTED_COVERAGE_MM = 340.0


def _make_subject(
    *,
    shape: tuple[int, int, int] = (12, 12, 12),
    left: tuple[int, int, int] = (2, 3, 4),
    right: tuple[int, int, int] = (8, 3, 10),
    affine: torch.Tensor | None = None,
) -> tio.Subject:
    label = torch.zeros((1, *shape), dtype=torch.int16)
    label[0, left[0], left[1], left[2]] = 1
    label[0, right[0], right[1], right[2]] = 2
    if affine is None:
        affine = torch.eye(4, dtype=torch.float32)
    return tio.Subject({"label_combined_femoral_head": tio.LabelMap(tensor=label, affine=affine)})


@pytest.mark.fast
def test_rejects_non_subject_input() -> None:
    transform = AutoBilateralHipPose()
    with pytest.raises(TypeError, match=r"torchio\.Subject"):
        transform({"label_combined_femoral_head": None})


@pytest.mark.fast
def test_requires_both_masks_present() -> None:
    subject = _make_subject()
    subject["label_combined_femoral_head"].data[0].masked_fill_(
        subject["label_combined_femoral_head"].data[0] == RIGHT_LABEL_ID,
        0,
    )
    transform = AutoBilateralHipPose()

    with pytest.raises(InvalidBilateralMaskError, match="Both left and right"):
        transform(subject)


@pytest.mark.fast
def test_sets_translation_to_world_midpoint() -> None:
    affine = torch.diag(torch.tensor([2.0, 3.0, 4.0, 1.0], dtype=torch.float32))
    subject = _make_subject(affine=affine)
    transform = AutoBilateralHipPose()

    transformed = transform(subject)

    assert torch.allclose(
        transformed["translations"],
        torch.tensor([[10.0, 809.0, 28.0]], dtype=torch.float32),
    )
    assert torch.equal(transformed["rotations"], torch.zeros((1, 3), dtype=torch.float32))


@pytest.mark.fast
def test_sets_dynamic_detector_spacing_from_world_bbox() -> None:
    affine = torch.diag(torch.tensor([10.0, 1.0, 10.0, 1.0], dtype=torch.float32))
    subject = _make_subject(shape=(50, 8, 50), left=(5, 3, 5), right=(35, 3, 35), affine=affine)
    transform = AutoBilateralHipPose()

    transformed = transform(subject)

    drr_config = transformed["drr_config"]
    expected_spacing = 340.0 * 1020.0 / (800.0 * 200.0)
    assert pytest.approx(drr_config["delx"], rel=1e-6) == expected_spacing
    assert pytest.approx(drr_config["dely"], rel=1e-6) == expected_spacing
    assert transformed["pose_metadata"]["coverage_x_mm"] == EXPECTED_COVERAGE_MM
    assert transformed["pose_metadata"]["coverage_z_mm"] == EXPECTED_COVERAGE_MM
