import numpy as np
import torch
import torchio as tio

from tazotron.datasets.transforms.femoral_head import AddFemoralHeadMasks

LEFT_LABEL_ID = 1
RIGHT_LABEL_ID = 2
EXPECTED_HEAD_X_MIN = 17
DIMENSIONS_3D = 3
DIMENSIONS_4D = 4
EXPECTED_ROUNDNESS_BACKSTEP_X_MIN = 25
EXPECTED_SINGLE_NOISE_X_MIN = 8


class _AddFemoralHeadMasksHarness(AddFemoralHeadMasks):
    def extract_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._extract_femoral_head_mask(tensor, affine=np.eye(4))


def _make_tapered_mask(
    shape: tuple[int, int, int],
    *,
    center_z: int,
    center_y: int,
    x_start: int,
    x_end: int,
) -> torch.Tensor:
    tensor = torch.zeros((1, *shape), dtype=torch.int16)
    for x in range(x_start, x_end):
        if x >= x_end - 8:
            radius = 6
        elif x >= x_end - 12:
            radius = 4
        else:
            radius = 2
        z_min = max(center_z - radius, 0)
        z_max = min(center_z + radius + 1, shape[0])
        y_min = max(center_y - radius, 0)
        y_max = min(center_y + radius + 1, shape[1])
        for z_idx in range(z_min, z_max):
            for y_idx in range(y_min, y_max):
                if (z_idx - center_z) ** 2 + (y_idx - center_y) ** 2 <= radius**2:
                    tensor[0, z_idx, y_idx, x] = 1
    return tensor


def _make_subject(
    left_mask: torch.Tensor,
    right_mask: torch.Tensor,
    *,
    with_combined: bool = True,
    with_outputs: bool = False,
    affine: np.ndarray | None = None,
) -> tio.Subject:
    shape = left_mask.shape[1:]
    label_affine = np.eye(4) if affine is None else affine
    data: dict[str, tio.Image] = {
        "volume": tio.ScalarImage(tensor=torch.zeros((1, *shape), dtype=torch.float32), affine=label_affine),
        "label_femur_left": tio.LabelMap(tensor=left_mask, affine=label_affine),
        "label_femur_right": tio.LabelMap(tensor=right_mask, affine=label_affine),
    }
    if with_combined:
        data["label_combined_femoral_head"] = tio.LabelMap(
            tensor=torch.zeros((1, *shape), dtype=torch.int16),
            affine=label_affine,
        )
    if with_outputs:
        data["label_femoral_head_left"] = tio.LabelMap(
            tensor=torch.ones((1, *shape), dtype=torch.int16),
            affine=label_affine,
        )
        data["label_femoral_head_right"] = tio.LabelMap(
            tensor=torch.full((1, *shape), RIGHT_LABEL_ID, dtype=torch.int16),
            affine=label_affine,
        )
    return tio.Subject(data)


def _make_roundness_transition_mask(  # noqa: PLR0913
    shape: tuple[int, int, int],
    *,
    center_z: int,
    center_y: int,
    x_start: int,
    x_end: int,
    bad_slice_start: int,
    bad_slice_end: int,
    round_radius: int = 6,
    elongated_z_radius: int = 8,
    elongated_y_radius: int = 3,
) -> torch.Tensor:
    tensor = torch.zeros((1, *shape), dtype=torch.int16)
    for x_idx in range(x_start, x_end):
        is_bad_slice = bad_slice_end <= x_idx <= bad_slice_start
        z_radius = elongated_z_radius if is_bad_slice else round_radius
        y_radius = elongated_y_radius if is_bad_slice else round_radius
        z_min = max(center_z - z_radius, 0)
        z_max = min(center_z + z_radius + 1, shape[0])
        y_min = max(center_y - y_radius, 0)
        y_max = min(center_y + y_radius + 1, shape[1])
        for z_idx in range(z_min, z_max):
            for y_idx in range(y_min, y_max):
                z_term = ((z_idx - center_z) / z_radius) ** 2
                y_term = ((y_idx - center_y) / y_radius) ** 2
                if z_term + y_term <= 1.0:
                    tensor[0, z_idx, y_idx, x_idx] = 1
    return tensor


def _x_min(mask: torch.Tensor) -> int:
    return int((mask[0] > 0).nonzero(as_tuple=False)[:, 2].min().item())


def _x_mean(mask: torch.Tensor) -> float:
    return float((mask[0] > 0).nonzero(as_tuple=False)[:, 2].to(torch.float64).mean().item())


def test_extracts_head_by_area_profile_on_tapered_shape() -> None:
    shape = (40, 40, 40)
    left_mask = _make_tapered_mask(shape, center_z=12, center_y=20, x_start=8, x_end=30)
    right_mask = _make_tapered_mask(shape, center_z=28, center_y=20, x_start=8, x_end=30)
    subject = _make_subject(left_mask, right_mask)

    transform = AddFemoralHeadMasks(area_ratio_threshold=0.55)
    subject = transform(subject)

    left = subject["label_femoral_head_left"].data
    right = subject["label_femoral_head_right"].data

    assert left.shape == left_mask.shape
    assert right.shape == right_mask.shape
    assert int(left.sum().item()) > 0
    assert int(right.sum().item()) > 0
    left_x_min = int((left[0] > 0).nonzero(as_tuple=False)[:, 2].min().item())
    right_x_min = int((right[0] > 0).nonzero(as_tuple=False)[:, 2].min().item())
    assert left_x_min >= EXPECTED_HEAD_X_MIN
    assert right_x_min >= EXPECTED_HEAD_X_MIN


def test_plane_cap_is_hard_cut() -> None:
    shape = (40, 40, 40)
    left_mask = _make_tapered_mask(shape, center_z=12, center_y=20, x_start=8, x_end=30)
    right_mask = _make_tapered_mask(shape, center_z=28, center_y=20, x_start=8, x_end=30)
    subject = _make_subject(left_mask, right_mask, with_combined=False)

    transform = AddFemoralHeadMasks(area_ratio_threshold=0.55)
    transformed = transform(subject)
    left = transformed["label_femoral_head_left"].data[0] > 0
    left_source = left_mask[0] > 0
    cutoff = int(left.nonzero(as_tuple=False)[:, 2].min().item())

    expected = torch.zeros_like(left)
    expected[:, :, cutoff:] = left_source[:, :, cutoff:]
    assert torch.equal(left, expected)


def test_respects_area_ratio_threshold() -> None:
    shape = (40, 40, 40)
    left_mask = _make_tapered_mask(shape, center_z=12, center_y=20, x_start=8, x_end=30)
    right_mask = _make_tapered_mask(shape, center_z=28, center_y=20, x_start=8, x_end=30)
    subject_low = _make_subject(left_mask.clone(), right_mask.clone(), with_combined=False)
    subject_high = _make_subject(left_mask, right_mask, with_combined=False)

    lower_threshold = AddFemoralHeadMasks(area_ratio_threshold=0.40)
    higher_threshold = AddFemoralHeadMasks(area_ratio_threshold=0.80)
    out_low = lower_threshold(subject_low)
    out_high = higher_threshold(subject_high)

    left_low_voxels = int((out_low["label_femoral_head_left"].data > 0).sum().item())
    left_high_voxels = int((out_high["label_femoral_head_left"].data > 0).sum().item())
    assert left_high_voxels < left_low_voxels


def test_handles_empty_mask() -> None:
    shape = (32, 32, 32)
    left_mask = torch.zeros((1, *shape), dtype=torch.int16)
    right_mask = torch.zeros((1, *shape), dtype=torch.int16)
    subject = _make_subject(left_mask, right_mask)

    transform = AddFemoralHeadMasks()
    transformed = transform(subject)
    assert int(transformed["label_femoral_head_left"].data.sum().item()) == 0
    assert int(transformed["label_femoral_head_right"].data.sum().item()) == 0


def test_keeps_largest_component_only() -> None:
    shape = (40, 40, 40)
    left_mask = _make_tapered_mask(shape, center_z=12, center_y=20, x_start=8, x_end=30)
    left_mask[0, 36:38, 36:38, 36:38] = 1
    right_mask = _make_tapered_mask(shape, center_z=28, center_y=20, x_start=8, x_end=30)
    subject = _make_subject(left_mask, right_mask, with_combined=False)

    transform = AddFemoralHeadMasks(min_component_voxels=1)
    transformed = transform(subject)
    left = transformed["label_femoral_head_left"].data
    assert int(left[0, 36:38, 36:38, 36:38].sum().item()) == 0


def test_updates_combined_label_left_right_ids() -> None:
    shape = (40, 40, 40)
    left_mask = _make_tapered_mask(shape, center_z=12, center_y=20, x_start=8, x_end=30)
    right_mask = _make_tapered_mask(shape, center_z=28, center_y=20, x_start=8, x_end=30)
    subject = _make_subject(left_mask, right_mask)

    transform = AddFemoralHeadMasks()
    transformed = transform(subject)

    combined = transformed["label_combined_femoral_head"].data
    assert torch.any(combined == LEFT_LABEL_ID)
    assert torch.any(combined == RIGHT_LABEL_ID)
    assert not torch.any((combined != 0) & (combined != LEFT_LABEL_ID) & (combined != RIGHT_LABEL_ID))


def test_overwrite_false_preserves_existing_outputs() -> None:
    shape = (40, 40, 40)
    left_mask = _make_tapered_mask(shape, center_z=12, center_y=20, x_start=8, x_end=30)
    right_mask = _make_tapered_mask(shape, center_z=28, center_y=20, x_start=8, x_end=30)
    subject = _make_subject(left_mask, right_mask, with_outputs=True)
    left_before = subject["label_femoral_head_left"].data.clone()
    right_before = subject["label_femoral_head_right"].data.clone()

    transform = AddFemoralHeadMasks(overwrite=False)
    transformed = transform(subject)
    assert torch.equal(transformed["label_femoral_head_left"].data, left_before)
    assert torch.equal(transformed["label_femoral_head_right"].data, right_before)


def test_supports_3d_and_4d_label_tensors() -> None:
    shape = (40, 40, 40)
    mask_4d = _make_tapered_mask(shape, center_z=12, center_y=20, x_start=8, x_end=30)
    mask_3d = mask_4d[0]
    transform = _AddFemoralHeadMasksHarness()

    out_4d = transform.extract_from_tensor(mask_4d)
    out_3d = transform.extract_from_tensor(mask_3d)
    assert out_4d.ndim == DIMENSIONS_4D
    assert out_3d.ndim == DIMENSIONS_3D
    assert int(out_4d.sum().item()) > 0
    assert int(out_3d.sum().item()) > 0


def test_roundness_triggers_earlier_cutoff_than_area() -> None:
    shape = (48, 48, 48)
    left_mask = _make_roundness_transition_mask(
        shape,
        center_z=14,
        center_y=24,
        x_start=8,
        x_end=40,
        bad_slice_start=24,
        bad_slice_end=8,
    )
    right_mask = _make_roundness_transition_mask(
        shape,
        center_z=34,
        center_y=24,
        x_start=8,
        x_end=40,
        bad_slice_start=24,
        bad_slice_end=8,
    )
    with_roundness = _make_subject(left_mask.clone(), right_mask.clone(), with_combined=False)
    area_only = _make_subject(left_mask, right_mask, with_combined=False)

    out_roundness = AddFemoralHeadMasks(
        area_ratio_threshold=0.2,
        roundness_ratio_threshold=1.2,
        roundness_confirm_slices=2,
        roundness_backstep_slices=1,
    )(with_roundness)
    out_area_only = AddFemoralHeadMasks(
        area_ratio_threshold=0.2,
        roundness_ratio_threshold=100.0,
    )(area_only)

    roundness_cutoff_x = _x_min(out_roundness["label_femoral_head_left"].data)
    area_cutoff_x = _x_min(out_area_only["label_femoral_head_left"].data)
    assert roundness_cutoff_x > area_cutoff_x


def test_roundness_backstep_keeps_last_good_slice() -> None:
    shape = (48, 48, 48)
    left_mask = _make_roundness_transition_mask(
        shape,
        center_z=14,
        center_y=24,
        x_start=8,
        x_end=40,
        bad_slice_start=24,
        bad_slice_end=8,
    )
    right_mask = _make_roundness_transition_mask(
        shape,
        center_z=34,
        center_y=24,
        x_start=8,
        x_end=40,
        bad_slice_start=24,
        bad_slice_end=8,
    )
    subject = _make_subject(left_mask, right_mask, with_combined=False)
    transformed = AddFemoralHeadMasks(
        area_ratio_threshold=0.2,
        roundness_ratio_threshold=1.2,
        roundness_confirm_slices=2,
        roundness_backstep_slices=1,
    )(subject)

    assert _x_min(transformed["label_femoral_head_left"].data) == EXPECTED_ROUNDNESS_BACKSTEP_X_MIN


def test_roundness_confirm_slices_ignores_single_noisy_slice() -> None:
    shape = (48, 48, 48)
    left_mask = _make_roundness_transition_mask(
        shape,
        center_z=14,
        center_y=24,
        x_start=8,
        x_end=40,
        bad_slice_start=24,
        bad_slice_end=24,
    )
    right_mask = _make_roundness_transition_mask(
        shape,
        center_z=34,
        center_y=24,
        x_start=8,
        x_end=40,
        bad_slice_start=24,
        bad_slice_end=24,
    )
    subject = _make_subject(left_mask, right_mask, with_combined=False)
    transformed = AddFemoralHeadMasks(
        area_ratio_threshold=0.2,
        roundness_ratio_threshold=1.2,
        roundness_confirm_slices=2,
        roundness_backstep_slices=1,
    )(subject)

    assert _x_min(transformed["label_femoral_head_left"].data) == EXPECTED_SINGLE_NOISE_X_MIN


def test_world_superior_orients_pca_sign() -> None:
    shape = (40, 40, 40)
    left_mask = _make_tapered_mask(shape, center_z=12, center_y=20, x_start=8, x_end=30)
    right_mask = _make_tapered_mask(shape, center_z=28, center_y=20, x_start=8, x_end=30)

    subject_identity = _make_subject(left_mask.clone(), right_mask.clone(), with_combined=False)
    transformed_identity = AddFemoralHeadMasks(area_ratio_threshold=0.55)(subject_identity)
    identity_mean_x = _x_mean(transformed_identity["label_femoral_head_left"].data)

    flipped_affine = np.eye(4)
    flipped_affine[2, 2] = -1.0
    subject_flipped = _make_subject(left_mask, right_mask, with_combined=False, affine=flipped_affine)
    transformed_flipped = AddFemoralHeadMasks(area_ratio_threshold=0.55)(subject_flipped)
    flipped_mean_x = _x_mean(transformed_flipped["label_femoral_head_left"].data)

    assert flipped_mean_x < identity_mean_x


def test_validates_roundness_parameters() -> None:
    with np.testing.assert_raises_regex(ValueError, "roundness_ratio_threshold"):
        AddFemoralHeadMasks(roundness_ratio_threshold=0.9)
    with np.testing.assert_raises_regex(ValueError, "roundness_confirm_slices"):
        AddFemoralHeadMasks(roundness_confirm_slices=0)
    with np.testing.assert_raises_regex(ValueError, "roundness_backstep_slices"):
        AddFemoralHeadMasks(roundness_backstep_slices=-1)
