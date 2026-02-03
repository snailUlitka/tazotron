import torch
import torchio as tio

from tazotron.datasets.transforms.femoral_head import AddFemoralHeadMasks


def _make_mask(shape: tuple[int, int, int], z_start: int, z_end: int) -> torch.Tensor:
    tensor = torch.zeros((1, *shape), dtype=torch.int16)
    tensor[0, z_start:z_end, :, :] = 1
    return tensor


def test_adds_femoral_head_masks_and_updates_label() -> None:
    shape = (32, 32, 32)
    left_mask = _make_mask(shape, 0, 16)
    right_mask = _make_mask(shape, 16, 32)

    subject = tio.Subject(
        volume=tio.ScalarImage(tensor=torch.zeros((1, *shape)), affine=torch.eye(4)),
        label_combined_femoral_head=tio.LabelMap(
            tensor=torch.zeros((1, *shape), dtype=torch.int16),
            affine=torch.eye(4),
        ),
        label_hip_left=tio.LabelMap(tensor=left_mask, affine=torch.eye(4)),
        label_hip_right=tio.LabelMap(tensor=right_mask, affine=torch.eye(4)),
    )

    transform = AddFemoralHeadMasks(cube_size=10, seed=0)
    subject = transform(subject)

    left = subject["label_femoral_head_left"].data
    right = subject["label_femoral_head_right"].data

    assert left.shape == left_mask.shape
    assert right.shape == right_mask.shape
    assert int(left.sum().item()) == 10**3
    assert int(right.sum().item()) == 10**3

    combined = subject["label_combined_femoral_head"].data
    assert torch.any(combined == 1)
    assert torch.any(combined == 2)
