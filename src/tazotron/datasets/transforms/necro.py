"""Data transformations for volumetric medical images."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torchvision.transforms import v2

if TYPE_CHECKING:
    import torchio as tio

CT_EXPECTED_DIMS = 4
LABEL_EXPECTED_DIMS = 3
LABEL_CHANNELS = 1
TARGET_LABEL = 1
NECROSIS_HU = 50.0


class AddRandomNecrosis(v2.Transform):
    """Inject necrosis into a femoral head mask by setting random voxels to a fixed HU value.

    This transform is tailored for subjects produced by `tazotron.datasets.ct.CTDataset`:
    it expects `subject["ct"]` and `subject["label"]`, and applies changes in place.

    Parameters
    ----------
    intensity:
        Fraction of voxels inside the target mask to replace with necrosis. Must be in [0, 1].
    seed:
        Optional seed for deterministic voxel selection.
    """

    def __init__(self, intensity: float = 0.1, seed: int | None = None) -> None:
        super().__init__()
        if not (0.0 <= intensity <= 1.0):
            msg = "intensity must be in the range [0, 1]."
            raise ValueError(msg)
        self.intensity = intensity
        self.seed = seed

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Apply the transform to a `torchio.Subject` in place."""
        ct_image = subject["ct"]
        label_image = subject["label"]

        ct = self._image_to_tensor(ct_image)
        label = self._image_to_tensor(label_image)
        mask = self._mask_from_label(label)
        if not mask.any() or self.intensity == 0.0:
            return subject

        updated_ct = self._apply_necrosis(ct, mask)
        self._assign_image_tensor(ct_image, updated_ct)
        return subject

    def _apply_necrosis(self, ct: Tensor, mask: Tensor) -> Tensor:
        if ct.ndim != CT_EXPECTED_DIMS:
            msg = "Expected CT tensor shaped (1, D, H, W)."
            raise ValueError(msg)

        working = ct
        coords = mask.nonzero(as_tuple=False)
        if coords.numel() == 0:
            return working

        num_mask_voxels = coords.shape[0]
        num_necrosis_voxels = int(num_mask_voxels * self.intensity)
        if num_necrosis_voxels <= 0:
            return working

        generator: torch.Generator | None = None
        if self.seed is not None:
            generator = torch.Generator(device=mask.device.type)
            generator.manual_seed(self.seed)

        perm = torch.randperm(
            num_mask_voxels,
            device=mask.device,
            generator=generator,
        )[:num_necrosis_voxels]
        selected = coords[perm]
        z, y, x = selected.unbind(dim=1)
        working[0, z, y, x] = NECROSIS_HU
        return working

    def _mask_from_label(self, label: Tensor) -> Tensor:
        if label.ndim == CT_EXPECTED_DIMS and label.shape[0] == LABEL_CHANNELS:
            label = label[0]
        if label.ndim != LABEL_EXPECTED_DIMS:
            msg = "Expected labelmap shaped (D, H, W) or (1, D, H, W)."
            raise ValueError(msg)
        return label == TARGET_LABEL

    def _image_to_tensor(self, image: tio.Image) -> Tensor:
        if hasattr(image, "data"):
            return image.data
        msg = "Failed to extract tensor from torchio.Image."
        raise TypeError(msg)

    def _assign_image_tensor(self, image: tio.Image, tensor: Tensor) -> None:
        if hasattr(image, "set_data"):
            image.set_data(tensor)
            return
        msg = "Failed to assign tensor to torchio.Image."
        raise TypeError(msg)
