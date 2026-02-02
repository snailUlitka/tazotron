"""Bilateral hip ROI crop that keeps both femoral heads in view."""

from __future__ import annotations

from typing import Any

import torch
import torchio as tio
from torch import Tensor
from torchvision.transforms import v2


class BilateralHipROICrop(v2.Transform):
    """Crop a bilateral hip ROI containing both femoral heads."""

    def __init__(
        self,
        label_name: str = "label",
        left_id: int = 1,
        right_id: int = 2,
        min_lr_mm: float = 240.0,
        margin_lr_mm: float = 20.0,
    ) -> None:
        super().__init__()
        self.label_name = label_name
        self.left_id = left_id
        self.right_id = right_id
        self.min_lr_mm = min_lr_mm
        self.margin_lr_mm = margin_lr_mm

    def __call__(self, subject: Any) -> Any:
        """Crop both femoral heads into a single ROI."""
        if not isinstance(subject, tio.Subject):
            msg = "BilateralHipROICrop expects a torchio.Subject."
            raise TypeError(msg)
        label_image = subject[self.label_name]
        masks = self._extract_masks(label_image)
        center_left, center_right = self._centers(masks)
        bounds = self._compute_bounds(
            centers=(center_left, center_right),
            masks=masks,
            spacing=label_image.spacing,
            shape=label_image.data.shape,
        )
        cropped_images: dict[str, tio.Image] = {}
        for name, img in subject.get_images_dict(intensity_only=False).items():
            cropped_images[name] = self._crop_image(img, bounds)
        extras = {key: value for key, value in subject.items() if key not in cropped_images}
        return tio.Subject({**cropped_images, **extras})

    def _extract_masks(self, label_image: tio.LabelMap) -> tuple[Tensor, Tensor]:
        data = label_image.data[0]
        left_mask = data == self.left_id
        right_mask = data == self.right_id
        if not left_mask.any() or not right_mask.any():
            msg = "Both left and right hip masks must be present for bilateral crop."
            raise ValueError(msg)
        return left_mask, right_mask

    def _centers(self, masks: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        left_mask, right_mask = masks
        center_left = left_mask.nonzero(as_tuple=False).float().mean(dim=0)
        center_right = right_mask.nonzero(as_tuple=False).float().mean(dim=0)
        return center_left, center_right

    def _compute_bounds(
        self,
        centers: tuple[Tensor, Tensor],
        masks: tuple[Tensor, Tensor],
        spacing: tuple[float, float, float],
        shape: torch.Size,
    ) -> tuple[int, int, int, int, int, int]:
        sx = spacing[2]
        center_left, center_right = centers
        lr_distance_mm = torch.abs(center_left[2] - center_right[2]) * sx
        size_lr_mm = torch.maximum(
            torch.tensor(self.min_lr_mm, device=center_left.device),
            lr_distance_mm + 2 * self.margin_lr_mm,
        )
        half_lr_vox = float((size_lr_mm / 2.0 / sx).item())
        union_mask = masks[0] | masks[1]
        indices = union_mask.nonzero(as_tuple=False)
        z_min = int(indices[:, 0].min().item())
        z_max = int(indices[:, 0].max().item())
        y_min = 0
        y_max = shape[-2] - 1
        cx = (center_left[2] + center_right[2]) / 2.0
        x_min = int(torch.round(cx - half_lr_vox).item())
        x_max = int(torch.round(cx + half_lr_vox).item())
        _, z_dim, y_dim, x_dim = shape
        z_min = max(z_min, 0)
        y_min = max(y_min, 0)
        x_min = max(x_min, 0)
        z_max = min(z_max, z_dim - 1)
        y_max = min(y_max, y_dim - 1)
        x_max = min(x_max, x_dim - 1)
        return z_min, z_max, y_min, y_max, x_min, x_max

    def _crop_image(self, image: tio.Image, bounds: tuple[int, int, int, int, int, int]) -> tio.Image:
        z_min, z_max, y_min, y_max, x_min, x_max = bounds
        data = image.data
        cropped = data[..., z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
        affine = image.affine.copy()
        offset_vox = torch.tensor([z_min, y_min, x_min], dtype=torch.float32)
        affine[:3, 3] += affine[:3, :3] @ offset_vox.numpy()
        return image.__class__(tensor=cropped, affine=affine)
