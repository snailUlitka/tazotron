"""Bilateral hip ROI crop that keeps both femoral heads in view."""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch
import torchio as tio
from torch import Tensor
from torchvision.transforms import v2


class BilateralHipROICrop(v2.Transform):
    """Crop a bilateral hip ROI containing both femoral heads."""

    def __init__(  # noqa: PLR0913
        self,
        label_name: str = "label",
        left_id: int = 1,
        right_id: int = 2,
        size_si_mm: float = 260.0,
        min_lr_mm: float = 240.0,
        margin_lr_mm: float = 20.0,
        size_ap_mm: float = 160.0,
        *,
        crop_depth: bool = False,
        axis_order: Literal["zyx", "zxy"] = "zyx",
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.label_name = label_name
        self.left_id = left_id
        self.right_id = right_id
        self.size_si_mm = size_si_mm
        self.min_lr_mm = min_lr_mm
        self.margin_lr_mm = margin_lr_mm
        self.size_ap_mm = size_ap_mm
        self.crop_depth = crop_depth
        self.inplace = inplace
        self.axis_order = axis_order

    def __call__(self, subject: Any) -> Any:
        """Crop both femoral heads into a single ROI."""
        if not isinstance(subject, tio.Subject):
            msg = "BilateralHipROICrop expects a torchio.Subject."
            raise TypeError(msg)
        working = subject if self.inplace else copy.deepcopy(subject)
        label_image = working[self.label_name]
        masks = self._extract_masks(label_image)
        center_left, center_right = self._centers(masks)
        center_vox = (center_left + center_right) / 2.0
        bounds = self._compute_bounds(
            center_vox,
            centers=(center_left, center_right),
            spacing=label_image.spacing,
            shape=label_image.data.shape,
            crop_depth=self.crop_depth,
        )
        cropped_images: dict[str, tio.Image] = {}
        for name, img in working.get_images_dict(intensity_only=False).items():
            cropped_images[name] = self._crop_image(img, bounds)
        extras = {k: v for k, v in working.items() if k not in cropped_images}
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
        center_vox: Tensor,
        centers: tuple[Tensor, Tensor],
        spacing: tuple[float, float, float],
        shape: torch.Size,
        *,
        crop_depth: bool,
    ) -> tuple[int, int, int, int, int, int]:
        sz, sy, sx = spacing
        center_left, center_right = centers
        lr_distance_mm = torch.abs(center_left[2] - center_right[2]) * sx
        size_lr_mm = torch.maximum(
            torch.tensor(self.min_lr_mm, device=center_vox.device),
            lr_distance_mm + 2 * self.margin_lr_mm,
        )
        half_sizes_vox = torch.tensor(
            [
                (self.size_si_mm / 2.0) / sz,
                (self.size_ap_mm / 2.0) / sy,
                (size_lr_mm / 2.0) / sx,
            ],
            dtype=torch.float32,
            device=center_vox.device,
        )
        cz, cy, cx = center_vox
        z_min = int(torch.round(cz - half_sizes_vox[0]).item())
        z_max = int(torch.round(cz + half_sizes_vox[0]).item())
        if crop_depth:
            y_min = int(torch.round(cy - half_sizes_vox[1]).item())
            y_max = int(torch.round(cy + half_sizes_vox[1]).item())
        else:
            y_min = 0
            y_max = shape[-2] - 1
        x_min = int(torch.round(cx - half_sizes_vox[2]).item())
        x_max = int(torch.round(cx + half_sizes_vox[2]).item())
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
