"""Transform to add femoral head masks derived from hip masks."""

from __future__ import annotations

import torch
import torchio as tio
from torch import Tensor
from torchvision.transforms import v2

from tazotron.datasets.ct import (
    COMBINED_FEMORAL_HEAD,
    FEMORAL_HEAD_LEFT,
    FEMORAL_HEAD_RIGHT,
    HIP_LEFT,
    HIP_RIGHT,
)

LABEL_EXPECTED_DIMS = 3
LABEL_CHANNELS = 1


class AddFemoralHeadMasks(v2.Transform):
    """Add femoral head labelmaps based on left/right hip masks.

    Expects a `torchio.Subject` that contains hip masks under `hip_left_key` and `hip_right_key`.
    Adds femoral head masks under `femoral_left_key` and `femoral_right_key`. Optionally updates
    the combined femoral head label entry when present.
    """

    def __init__(
        self,
        *,
        hip_left_key: str = HIP_LEFT,
        hip_right_key: str = HIP_RIGHT,
        femoral_left_key: str = FEMORAL_HEAD_LEFT,
        femoral_right_key: str = FEMORAL_HEAD_RIGHT,
        cube_size: int = 20,
        update_combined_label: bool = True,
        overwrite: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if cube_size <= 0:
            msg = "cube_size must be a positive integer."
            raise ValueError(msg)
        if seed is not None and generator is not None:
            msg = "seed and generator cannot be set at the same time."
            raise ValueError(msg)
        self.hip_left_key = hip_left_key
        self.hip_right_key = hip_right_key
        self.femoral_left_key = femoral_left_key
        self.femoral_right_key = femoral_right_key
        self.cube_size = cube_size
        self.update_combined_label = update_combined_label
        self.overwrite = overwrite
        self.seed = seed
        self._user_generator = generator
        self._seeded_generator: torch.Generator | None = None
        self._seeded_generator_device: torch.device | None = None

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        if not isinstance(subject, tio.Subject):
            msg = "AddFemoralHeadMasks expects a torchio.Subject."
            raise TypeError(msg)

        hip_left = subject.get(self.hip_left_key)
        hip_right = subject.get(self.hip_right_key)
        if hip_left is None or hip_right is None:
            msg = "Both hip masks must be present to derive femoral head labels."
            raise ValueError(msg)
        if not isinstance(hip_left, tio.LabelMap) or not isinstance(hip_right, tio.LabelMap):
            msg = "Hip masks must be torchio.LabelMap instances."
            raise TypeError(msg)
        if hip_left.data.shape != hip_right.data.shape:
            msg = "Hip mask shapes must match."
            raise ValueError(msg)

        left_existing = subject.get(self.femoral_left_key)
        right_existing = subject.get(self.femoral_right_key)
        if not self.overwrite and left_existing is not None and right_existing is not None:
            return subject

        femoral_left, femoral_right = self._infer_femoral_head_masks(hip_left, hip_right)
        if self.overwrite or left_existing is None:
            subject[self.femoral_left_key] = femoral_left
        if self.overwrite or right_existing is None:
            subject[self.femoral_right_key] = femoral_right

        if self.update_combined_label and COMBINED_FEMORAL_HEAD in subject and self.overwrite:
            label_image = subject[COMBINED_FEMORAL_HEAD]
            if isinstance(label_image, tio.LabelMap):
                updated = self._combine_labels(femoral_left, femoral_right, label_image)
                label_image.set_data(updated)

        return subject

    def _infer_femoral_head_masks(
        self,
        hip_left: tio.LabelMap,
        hip_right: tio.LabelMap,
    ) -> tuple[tio.LabelMap, tio.LabelMap]:
        """Return femoral head masks for left/right hips based on hip labelmaps."""
        left_tensor = self._random_cube_mask(
            hip_left.data,
            generator=self._get_generator(hip_left.data.device),
        )
        right_tensor = self._random_cube_mask(
            hip_right.data,
            generator=self._get_generator(hip_right.data.device),
        )
        return (
            tio.LabelMap(tensor=left_tensor, affine=hip_left.affine),
            tio.LabelMap(tensor=right_tensor, affine=hip_right.affine),
        )

    def _random_cube_mask(self, hip_tensor: Tensor, *, generator: torch.Generator | None) -> Tensor:
        mask, output = self._extract_mask(hip_tensor)
        if not mask.any():
            return output

        coords = mask.nonzero(as_tuple=False)
        choice = torch.randint(
            coords.shape[0],
            (1,),
            generator=generator,
            device=coords.device,
        ).item()
        center = coords[choice]

        depth, height, width = mask.shape
        cube_z = min(self.cube_size, depth)
        cube_y = min(self.cube_size, height)
        cube_x = min(self.cube_size, width)

        z0 = self._clamp_start(int(center[0].item()), cube_z, depth)
        y0 = self._clamp_start(int(center[1].item()), cube_y, height)
        x0 = self._clamp_start(int(center[2].item()), cube_x, width)

        if output.ndim == LABEL_EXPECTED_DIMS:
            output[z0 : z0 + cube_z, y0 : y0 + cube_y, x0 : x0 + cube_x] = 1
        else:
            output[0, z0 : z0 + cube_z, y0 : y0 + cube_y, x0 : x0 + cube_x] = 1
        return output

    def _extract_mask(self, hip_tensor: Tensor) -> tuple[Tensor, Tensor]:
        if hip_tensor.ndim == LABEL_EXPECTED_DIMS:
            mask = hip_tensor > 0
            output = torch.zeros_like(hip_tensor)
            return mask, output
        if hip_tensor.ndim == LABEL_EXPECTED_DIMS + 1 and hip_tensor.shape[0] == LABEL_CHANNELS:
            mask = hip_tensor[0] > 0
            output = torch.zeros_like(hip_tensor)
            return mask, output
        msg = "Hip mask must have shape (D, H, W) or (1, D, H, W)."
        raise ValueError(msg)

    def _clamp_start(self, center: int, cube: int, size: int) -> int:
        start = center - cube // 2
        if start < 0:
            return 0
        if start > size - cube:
            return size - cube
        return start

    def _combine_labels(
        self,
        femoral_left: tio.LabelMap,
        femoral_right: tio.LabelMap,
        base_label: tio.LabelMap,
    ) -> Tensor:
        if (
            femoral_left.data.shape != base_label.data.shape
            or femoral_right.data.shape != base_label.data.shape
        ):
            msg = "Femoral head mask shapes must match the combined label."
            raise ValueError(msg)
        combined = torch.zeros_like(base_label.data)
        combined = torch.where(femoral_left.data > 0, torch.ones_like(combined), combined)
        combined = torch.where(femoral_right.data > 0, torch.full_like(combined, 2), combined)
        return combined

    def _get_generator(self, device: torch.device) -> torch.Generator | None:
        if self._user_generator is not None:
            generator_device = getattr(self._user_generator, "device", device)
            if generator_device != device:
                msg = (
                    "Provided generator device does not match the hip mask device. "
                    "Create the generator on the same device as the tensors."
                )
                raise ValueError(msg)
            return self._user_generator

        if self.seed is None:
            return None

        if self._seeded_generator is None or self._seeded_generator_device != device:
            seeded = torch.Generator(device=device)
            seeded.manual_seed(self.seed)
            self._seeded_generator = seeded
            self._seeded_generator_device = device

        return self._seeded_generator
