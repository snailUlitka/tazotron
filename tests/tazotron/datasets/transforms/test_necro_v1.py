from __future__ import annotations

import copy

import pytest
import torch
import torchio as tio
from diffdrr.data import transform_hu_to_density
from torch.nn import functional

from tazotron.datasets.ct import COMBINED_FEMORAL_HEAD
from tazotron.datasets.transforms.necro import AddLateAVNLikeNecrosisV1

LEFT_LABEL_ID = 1
RIGHT_LABEL_ID = 2
SHAPE = (48, 48, 48)
HEAD_RADIUS = 7.0
DELTA_EPS = 1e-5
SMOOTHNESS_DELTA_EPS = 0.75
MIN_CONNECTED_RATIO = 0.85


def _make_sphere_mask(center: tuple[float, float, float], radius: float = HEAD_RADIUS) -> torch.Tensor:
    axis_0 = torch.arange(SHAPE[0], dtype=torch.float32)
    axis_1 = torch.arange(SHAPE[1], dtype=torch.float32)
    axis_2 = torch.arange(SHAPE[2], dtype=torch.float32)
    grid_0, grid_1, grid_2 = torch.meshgrid(axis_0, axis_1, axis_2, indexing="ij")
    distance_sq = (grid_0 - center[0]) ** 2 + (grid_1 - center[1]) ** 2 + (grid_2 - center[2]) ** 2
    return distance_sq <= radius**2


def _make_ct_tensor(left_mask: torch.Tensor, right_mask: torch.Tensor) -> torch.Tensor:
    axis_0 = torch.arange(SHAPE[0], dtype=torch.float32)
    axis_1 = torch.arange(SHAPE[1], dtype=torch.float32)
    axis_2 = torch.arange(SHAPE[2], dtype=torch.float32)
    grid_0, grid_1, grid_2 = torch.meshgrid(axis_0, axis_1, axis_2, indexing="ij")
    ct = -850.0 + 7.5 * grid_0 + 2.0 * grid_1 + 1.5 * grid_2
    ct = ct.clone()
    ct[left_mask] = 220.0 + 6.0 * grid_0[left_mask] - 2.0 * grid_1[left_mask] + 3.0 * grid_2[left_mask]
    ct[right_mask] = 250.0 + 5.0 * grid_0[right_mask] - 3.0 * grid_1[right_mask] + 2.5 * grid_2[right_mask]
    return ct.unsqueeze(0)


def _make_subject(
    *,
    include_right_head: bool = True,
    include_density: bool = False,
) -> tio.Subject:
    left_mask = _make_sphere_mask((14.0, 24.0, 24.0))
    right_mask = _make_sphere_mask((34.0, 24.0, 24.0)) if include_right_head else torch.zeros(SHAPE, dtype=torch.bool)
    combined = torch.zeros((1, *SHAPE), dtype=torch.int16)
    combined[0][left_mask] = LEFT_LABEL_ID
    combined[0][right_mask] = RIGHT_LABEL_ID
    ct = _make_ct_tensor(left_mask, right_mask)
    affine = torch.diag(torch.tensor([1.5, 1.5, 1.5, 1.0], dtype=torch.float32))
    data: dict[str, tio.Image] = {
        "volume": tio.ScalarImage(tensor=ct, affine=affine),
        COMBINED_FEMORAL_HEAD: tio.LabelMap(tensor=combined, affine=affine),
    }
    if include_density:
        density = transform_hu_to_density(ct, 1.0)
        data["density"] = tio.ScalarImage(tensor=density, affine=affine)
    return tio.Subject(data)


def _make_symmetric_subject() -> tio.Subject:
    left_mask = _make_sphere_mask((14.0, 24.0, 24.0))
    right_mask = _make_sphere_mask((34.0, 24.0, 24.0))
    combined = torch.zeros((1, *SHAPE), dtype=torch.int16)
    combined[0][left_mask] = LEFT_LABEL_ID
    combined[0][right_mask] = RIGHT_LABEL_ID

    axis_0 = torch.arange(SHAPE[0], dtype=torch.float32)
    axis_1 = torch.arange(SHAPE[1], dtype=torch.float32)
    axis_2 = torch.arange(SHAPE[2], dtype=torch.float32)
    grid_0, grid_1, grid_2 = torch.meshgrid(axis_0, axis_1, axis_2, indexing="ij")
    mirrored_axis_0 = torch.minimum(grid_0, torch.full_like(grid_0, SHAPE[0] - 1) - grid_0)
    radial_left = torch.sqrt((grid_0 - 14.0) ** 2 + (grid_1 - 24.0) ** 2 + (grid_2 - 24.0) ** 2)
    radial_right = torch.sqrt((grid_0 - 34.0) ** 2 + (grid_1 - 24.0) ** 2 + (grid_2 - 24.0) ** 2)
    ct = -850.0 + 5.0 * mirrored_axis_0 + 1.5 * grid_1 + 1.5 * grid_2
    ct = ct.clone()
    ct[left_mask] = 320.0 - 8.0 * radial_left[left_mask] + 1.5 * grid_2[left_mask]
    ct[right_mask] = 320.0 - 8.0 * radial_right[right_mask] + 1.5 * grid_2[right_mask]
    affine = torch.diag(torch.tensor([1.5, 1.5, 1.5, 1.0], dtype=torch.float32))
    return tio.Subject(
        {
            "volume": tio.ScalarImage(tensor=ct.unsqueeze(0), affine=affine),
            COMBINED_FEMORAL_HEAD: tio.LabelMap(tensor=combined, affine=affine),
        },
    )


def _delta_inside_head(subject: tio.Subject, original_ct: torch.Tensor, label_id: int) -> torch.Tensor:
    delta = subject["volume"].data - original_ct
    label = subject[COMBINED_FEMORAL_HEAD].data
    mask = label == label_id
    return delta[mask]


@pytest.mark.fast
def test_valid_subject_runs_in_place() -> None:
    subject = _make_subject()
    original_id = id(subject)

    returned = AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "moderate", "seed": 7})(subject)

    assert returned is subject
    assert id(returned) == original_id


@pytest.mark.fast
def test_probability_one_changes_ct() -> None:
    subject = _make_subject()
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "moderate", "seed": 7})(subject)

    assert not torch.equal(subject["volume"].data, before)


@pytest.mark.fast
def test_probability_zero_leaves_ct_unchanged() -> None:
    subject = _make_subject()
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"probability": 0.0, "target_head": "left", "seed": 7})(subject)

    assert torch.equal(subject["volume"].data, before)


@pytest.mark.fast
def test_empty_mask_leaves_ct_unchanged() -> None:
    subject = _make_subject()
    subject[COMBINED_FEMORAL_HEAD].set_data(torch.zeros((1, *SHAPE), dtype=torch.int16))
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "moderate", "seed": 7})(subject)

    assert torch.equal(subject["volume"].data, before)


@pytest.mark.fast
def test_changes_stay_inside_selected_head() -> None:
    subject = _make_subject()
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "moderate", "seed": 7})(subject)

    label = subject[COMBINED_FEMORAL_HEAD].data
    delta = subject["volume"].data - before
    changed = delta.abs() > DELTA_EPS
    left_mask = label == LEFT_LABEL_ID
    assert torch.any(changed & left_mask)
    assert not torch.any(changed & ~left_mask)


@pytest.mark.fast
def test_opposite_head_is_unchanged() -> None:
    subject = _make_subject()
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "moderate", "seed": 7})(subject)

    label = subject[COMBINED_FEMORAL_HEAD].data
    right_mask = label == RIGHT_LABEL_ID
    assert torch.equal(subject["volume"].data[right_mask], before[right_mask])


@pytest.mark.fast
def test_random_target_modifies_exactly_one_available_head() -> None:
    subject = _make_subject()
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1(
        {
            "target_head": "random",
            "target_head_weights": (1.0, 0.0, 0.0),
            "severity": "moderate",
            "seed": 13,
        },
    )(subject)

    label = subject[COMBINED_FEMORAL_HEAD].data
    delta = (subject["volume"].data - before).abs()
    left_changed = bool(torch.any(delta[label == LEFT_LABEL_ID] > DELTA_EPS))
    right_changed = bool(torch.any(delta[label == RIGHT_LABEL_ID] > DELTA_EPS))
    assert left_changed != right_changed


@pytest.mark.fast
def test_both_target_modifies_both_heads_only() -> None:
    subject = _make_subject()
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "both", "severity": "moderate", "seed": 13})(subject)

    label = subject[COMBINED_FEMORAL_HEAD].data
    delta = (subject["volume"].data - before).abs()
    left_mask = label == LEFT_LABEL_ID
    right_mask = label == RIGHT_LABEL_ID

    assert torch.any(delta[left_mask] > DELTA_EPS)
    assert torch.any(delta[right_mask] > DELTA_EPS)
    assert not torch.any(delta[(~left_mask) & (~right_mask)] > DELTA_EPS)


@pytest.mark.fast
def test_bilateral_pattern_is_reproducible_for_same_seed() -> None:
    subject_a = _make_subject()
    subject_b = _make_subject()

    AddLateAVNLikeNecrosisV1({"target_head": "both", "severity": "random", "seed": 77})(subject_a)
    AddLateAVNLikeNecrosisV1({"target_head": "both", "severity": "random", "seed": 77})(subject_b)

    assert torch.equal(subject_a["volume"].data, subject_b["volume"].data)


@pytest.mark.fast
def test_bilateral_patterns_are_not_identical_between_sides() -> None:
    subject = _make_symmetric_subject()
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "both", "severity": "random", "seed": 41})(subject)

    label = subject[COMBINED_FEMORAL_HEAD].data
    delta = subject["volume"].data - before
    left_delta = delta[0] * (label[0] == LEFT_LABEL_ID)
    right_delta = delta[0] * (label[0] == RIGHT_LABEL_ID)
    mirrored_right_delta = torch.flip(right_delta, dims=(0,))

    assert not torch.allclose(left_delta, mirrored_right_delta, atol=1e-4, rtol=0.0)


@pytest.mark.fast
def test_missing_requested_side_is_noop() -> None:
    subject = _make_subject(include_right_head=False)
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "right", "severity": "moderate", "seed": 7})(subject)

    assert torch.equal(subject["volume"].data, before)


@pytest.mark.fast
def test_seed_reproduces_identical_sequence() -> None:
    subject_a1 = _make_subject()
    subject_a2 = _make_subject()
    subject_b1 = _make_subject()
    subject_b2 = _make_subject()

    transform_a = AddLateAVNLikeNecrosisV1({"target_head": "random", "severity": "moderate", "seed": 42})
    transform_b = AddLateAVNLikeNecrosisV1({"target_head": "random", "severity": "moderate", "seed": 42})

    transform_a(subject_a1)
    transform_a(subject_a2)
    transform_b(subject_b1)
    transform_b(subject_b2)

    assert torch.equal(subject_a1["volume"].data, subject_b1["volume"].data)
    assert torch.equal(subject_a2["volume"].data, subject_b2["volume"].data)


@pytest.mark.fast
def test_reused_transform_produces_reproducible_but_non_identical_sequence() -> None:
    subject_a1 = _make_subject()
    subject_a2 = _make_subject(include_right_head=False)
    subject_b1 = _make_subject()
    subject_b2 = _make_subject(include_right_head=False)

    transform_a = AddLateAVNLikeNecrosisV1({"target_head": "random", "severity": "random", "seed": 42})
    transform_b = AddLateAVNLikeNecrosisV1({"target_head": "random", "severity": "random", "seed": 42})

    transform_a(subject_a1)
    transform_a(subject_a2)
    transform_b(subject_b1)
    transform_b(subject_b2)

    assert torch.equal(subject_a1["volume"].data, subject_b1["volume"].data)
    assert torch.equal(subject_a2["volume"].data, subject_b2["volume"].data)
    assert not torch.equal(subject_a1["volume"].data, subject_a2["volume"].data)


@pytest.mark.fast
def test_random_target_weights_can_force_bilateral_mode() -> None:
    subject = _make_subject()
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1(
        {
            "target_head": "random",
            "target_head_weights": (0.0, 0.0, 1.0),
            "severity": "moderate",
            "seed": 17,
        },
    )(subject)

    label = subject[COMBINED_FEMORAL_HEAD].data
    delta = (subject["volume"].data - before).abs()
    assert torch.any(delta[label == LEFT_LABEL_ID] > DELTA_EPS)
    assert torch.any(delta[label == RIGHT_LABEL_ID] > DELTA_EPS)


@pytest.mark.fast
def test_random_severity_is_reproducible_for_same_seed() -> None:
    subject_a = _make_subject()
    subject_b = _make_subject()

    AddLateAVNLikeNecrosisV1({"target_head": "random", "severity": "random", "seed": 101})(subject_a)
    AddLateAVNLikeNecrosisV1({"target_head": "random", "severity": "random", "seed": 101})(subject_b)

    assert torch.equal(subject_a["volume"].data, subject_b["volume"].data)


@pytest.mark.fast
def test_different_seeds_produce_different_patterns() -> None:
    subject_a = _make_subject()
    subject_b = _make_subject()

    AddLateAVNLikeNecrosisV1({"target_head": "random", "severity": "random", "seed": 21})(subject_a)
    AddLateAVNLikeNecrosisV1({"target_head": "random", "severity": "random", "seed": 22})(subject_b)

    assert not torch.equal(subject_a["volume"].data, subject_b["volume"].data)


@pytest.mark.fast
def test_probability_gate_is_seeded_and_can_skip() -> None:
    seed = 123
    probability = 0.2
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    expected_apply = float(torch.rand((), generator=generator).item()) <= probability

    subject = _make_subject()
    before = subject["volume"].data.clone()
    AddLateAVNLikeNecrosisV1(
        {"probability": probability, "target_head": "left", "severity": "moderate", "seed": seed},
    )(subject)

    changed = not torch.equal(subject["volume"].data, before)
    assert changed is expected_apply


@pytest.mark.fast
def test_density_is_recomputed_when_present() -> None:
    subject = _make_subject(include_density=True)
    old_density = subject["density"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "moderate", "seed": 7})(subject)

    assert not torch.equal(subject["density"].data, old_density)
    expected_density = transform_hu_to_density(subject["volume"].data, 1.0)
    assert torch.allclose(subject["density"].data, expected_density)


@pytest.mark.fast
def test_severity_levels_increase_mean_effect() -> None:
    subject_mild = _make_subject()
    subject_moderate = _make_subject()
    subject_severe = _make_subject()
    base_ct = subject_mild["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "mild", "seed": 11})(subject_mild)
    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "moderate", "seed": 11})(subject_moderate)
    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "severe", "seed": 11})(subject_severe)

    mild_delta = _delta_inside_head(subject_mild, base_ct, LEFT_LABEL_ID).abs().mean().item()
    moderate_delta = _delta_inside_head(subject_moderate, base_ct, LEFT_LABEL_ID).abs().mean().item()
    severe_delta = _delta_inside_head(subject_severe, base_ct, LEFT_LABEL_ID).abs().mean().item()
    assert mild_delta < moderate_delta < severe_delta


@pytest.mark.fast
def test_changed_pattern_is_smooth_not_sparse_noise() -> None:
    subject = _make_subject()
    before = subject["volume"].data.clone()

    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "severe", "seed": 29})(subject)

    label = subject[COMBINED_FEMORAL_HEAD].data
    left_mask = label == LEFT_LABEL_ID
    changed_mask = ((subject["volume"].data - before).abs() > SMOOTHNESS_DELTA_EPS) & left_mask
    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
    kernel[0, 0, 0, 1, 1] = 1.0
    kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = 1.0
    kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = 1.0
    kernel[0, 0, 1, 1, 2] = 1.0

    neighbors = functional.conv3d(changed_mask[None].to(torch.float32), kernel, padding=1)
    connected_ratio = neighbors[changed_mask[None]].gt(0).to(torch.float32).mean().item()
    assert connected_ratio > MIN_CONNECTED_RATIO


@pytest.mark.fast
def test_generator_can_be_provided_explicitly() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(99)
    subject = _make_subject()
    subject_copy = copy.deepcopy(subject)

    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "moderate", "generator": generator})(subject)
    generator_2 = torch.Generator(device="cpu")
    generator_2.manual_seed(99)
    AddLateAVNLikeNecrosisV1({"target_head": "left", "severity": "moderate", "generator": generator_2})(subject_copy)

    assert torch.equal(subject["volume"].data, subject_copy["volume"].data)
