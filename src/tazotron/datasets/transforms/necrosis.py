"""Data transformations for volumetric medical images."""

from __future__ import annotations

# ruff: noqa: PLR0913
import copy
from typing import Any

import torch
import torchio as tio
from torch import Tensor
from torchvision.transforms import v2

CT_EXPECTED_DIMS = 4
LABEL_EXPECTED_DIMS = 3
LABEL_CHANNELS = 1
CYST_PROBABILITY = 0.35


class AddRandomNecrosis(v2.Transform):
    """Injects random necrotic-like spots inside a femoral head mask.

    Supported input: `torchio.Subject` with CT and label stored under `ct_key` and `label_key`.

    Parameters
    ----------
    probability:
        Probability of applying the transform.
    target_label:
        Label value in the labelmap representing the femoral head.
    radius_range:
        Inclusive voxel radius range for each necrotic spot.
    intensity_drop:
        Fractional signal attenuation range at the spot center (0..1).
    blur_sigma:
        Gaussian sigma for edge smoothing of each spot. Zero yields a sharp boundary.
    num_spots:
        Inclusive range of necrotic spots per invocation.
    ct_key:
        CT volume key in a `torchio.Subject`.
    label_key:
        Labelmap key pointing to the femoral head mask.
    inplace:
        When `True`, mutates the input container; otherwise returns a copy.
    """

    def __init__(
        self,
        probability: float = 0.5,
        target_label: int = 1,
        target_labels: tuple[int, ...] | None = None,
        radius_range: tuple[int, int] = (80, 160),
        intensity_drop: tuple[float, float] = (0.0, 0.15),
        blur_sigma: float = 1.0,
        num_spots: tuple[int, int] = (1, 3),
        ct_key: str = "ct",
        label_key: str = "label",
        *,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        if not (0 <= probability <= 1):
            msg = "probability must be in the range [0, 1]."
            raise ValueError(msg)
        if radius_range[0] <= 0 or radius_range[0] > radius_range[1]:
            msg = "radius_range must be positive and ordered."
            raise ValueError(msg)
        if not (0 <= intensity_drop[0] <= intensity_drop[1] <= 1):
            msg = "intensity_drop must be within [0, 1] and ordered."
            raise ValueError(msg)
        if blur_sigma < 0:
            msg = "blur_sigma cannot be negative."
            raise ValueError(msg)
        if num_spots[0] <= 0 or num_spots[0] > num_spots[1]:
            msg = "num_spots must be positive and ordered."
            raise ValueError(msg)
        if target_labels is not None and len(target_labels) == 0:
            msg = "target_labels cannot be empty."
            raise ValueError(msg)
        self.probability = probability
        self.target_label = target_label
        self.target_labels = target_labels
        self.radius_range = radius_range
        self.intensity_drop = intensity_drop
        self.blur_sigma = blur_sigma
        self.num_spots = num_spots
        self.ct_key = ct_key
        self.label_key = label_key
        self.inplace = inplace

    def __call__(self, subject: Any) -> Any:
        """Apply the transform to a `torchio.Subject`."""
        if torch.rand(1).item() > self.probability:
            return subject
        if not isinstance(subject, tio.Subject):
            msg = "Expected a torchio.Subject as input."
            raise TypeError(msg)
        result = subject if self.inplace else copy.deepcopy(subject)
        ct_image = result[self.ct_key]
        label_image = result[self.label_key]
        ct = self._image_to_tensor(ct_image)
        label = self._image_to_tensor(label_image)
        mask = self._mask_from_label(label)
        if not mask.any():
            return result
        updated_ct = self._apply_necrosis(ct, mask)
        self._assign_image_tensor(ct_image, updated_ct)
        return result

    def _apply_necrosis(self, ct: Tensor, mask: Tensor) -> Tensor:
        if ct.ndim != CT_EXPECTED_DIMS:
            msg = "Expected CT tensor shaped (1, D, H, W)."
            raise ValueError(msg)
        working = ct if self.inplace else ct.clone()
        coords = mask.nonzero(as_tuple=False)
        if coords.numel() == 0:
            return working
        spots = int(
            torch.randint(
                low=self.num_spots[0],
                high=self.num_spots[1] + 1,
                size=(1,),
                device=mask.device,
            ).item(),
        )
        for _ in range(spots):
            center = coords[
                torch.randint(
                    low=0,
                    high=coords.shape[0],
                    size=(1,),
                    device=mask.device,
                ).item()
            ]
            radius = int(
                torch.randint(
                    low=self.radius_range[0],
                    high=self.radius_range[1] + 1,
                    size=(1,),
                    device=mask.device,
                ).item(),
            )
            rim_thickness = max(1, min(6, int(max(1, radius * 0.1))))
            inner_radius = max(1, radius - rim_thickness)
            core_scale = 1.0 - float(
                torch.empty(1, device=mask.device).uniform_(*self.intensity_drop).item(),
            )
            rim_scale = float(torch.empty(1, device=mask.device).uniform_(1.5, 2.0).item())
            cyst_scale = float(torch.empty(1, device=mask.device).uniform_(0.0, 0.25).item())
            apply_cyst = torch.rand(1, device=mask.device).item() < CYST_PROBABILITY

            z0 = max(0, int(center[0].item() - radius))
            y0 = max(0, int(center[1].item() - radius))
            x0 = max(0, int(center[2].item() - radius))
            z1 = min(mask.shape[0] - 1, int(center[0].item() + radius))
            y1 = min(mask.shape[1] - 1, int(center[1].item() + radius))
            x1 = min(mask.shape[2] - 1, int(center[2].item() + radius))

            region_mask = mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
            if not region_mask.any():
                continue

            z = torch.arange(z0, z1 + 1, device=mask.device, dtype=torch.float32) - center[0].float()
            y = torch.arange(y0, y1 + 1, device=mask.device, dtype=torch.float32) - center[1].float()
            x = torch.arange(x0, x1 + 1, device=mask.device, dtype=torch.float32) - center[2].float()
            zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
            dist = torch.sqrt(zz.square() + yy.square() + xx.square())

            core_mask = (dist <= inner_radius) & region_mask
            rim_mask = (dist > inner_radius) & (dist <= radius) & region_mask
            cyst_mask = torch.zeros_like(core_mask)
            if apply_cyst and core_mask.any():
                cyst_radius = max(1, int(radius * 0.25))
                cyst_center = core_mask.nonzero(as_tuple=False)[
                    torch.randint(
                        low=0,
                        high=core_mask.nonzero(as_tuple=False).shape[0],
                        size=(1,),
                        device=mask.device,
                    ).item()
                ]
                cz, cy, cx = cyst_center
                zc = torch.arange(z0, z1 + 1, device=mask.device, dtype=torch.float32) - (z0 + cz).float()
                yc = torch.arange(y0, y1 + 1, device=mask.device, dtype=torch.float32) - (y0 + cy).float()
                xc = torch.arange(x0, x1 + 1, device=mask.device, dtype=torch.float32) - (x0 + cx).float()
                zz_c, yy_c, xx_c = torch.meshgrid(zc, yc, xc, indexing="ij")
                cyst_dist = torch.sqrt(zz_c.square() + yy_c.square() + xx_c.square())
                cyst_mask = (cyst_dist <= cyst_radius) & core_mask

            scale_map = torch.ones_like(region_mask, dtype=working.dtype)
            scale_map = torch.where(core_mask, core_scale, scale_map)
            scale_map = torch.where(rim_mask, rim_scale, scale_map)
            if apply_cyst and cyst_mask.any():
                scale_map = torch.where(cyst_mask, cyst_scale, scale_map)

            working[
                :,
                z0 : z1 + 1,
                y0 : y1 + 1,
                x0 : x1 + 1,
            ] = working[
                :,
                z0 : z1 + 1,
                y0 : y1 + 1,
                x0 : x1 + 1,
            ] * scale_map.unsqueeze(0)
        return working

    def _mask_from_label(self, label: Tensor) -> Tensor:
        if label.ndim == CT_EXPECTED_DIMS and label.shape[0] == LABEL_CHANNELS:
            label = label[0]
        if label.ndim != LABEL_EXPECTED_DIMS:
            msg = "Expected labelmap shaped (D, H, W) or (1, D, H, W)."
            raise ValueError(msg)
        if self.target_labels is not None:
            mask = torch.zeros_like(label, dtype=torch.bool)
            for value in self.target_labels:
                mask |= label == value
            return mask
        return label == self.target_label

    def _image_to_tensor(self, image: Any) -> Tensor:
        if hasattr(image, "data"):
            return image.data
        msg = "Failed to extract tensor from torchio.Image."
        raise TypeError(msg)

    def _assign_image_tensor(self, image: Any, tensor: Tensor) -> None:
        if hasattr(image, "data"):
            image.data = tensor
            return
        msg = "Failed to assign tensor to torchio.Image."
        raise TypeError(msg)

    def _spherical_attenuation(
        self,
        shape: torch.Size,
        center: Tensor,
        radius: int,
        sigma: float,
        device: torch.device,
    ) -> Tensor:
        z = torch.arange(shape[0], device=device, dtype=torch.float32) - center[0].float()
        y = torch.arange(shape[1], device=device, dtype=torch.float32) - center[1].float()
        x = torch.arange(shape[2], device=device, dtype=torch.float32) - center[2].float()
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        dist_sq = zz.square() + yy.square() + xx.square()
        sphere = dist_sq <= float(radius**2)
        if sigma > 0:
            weight = torch.exp(-dist_sq / (2 * sigma * sigma))
        else:
            weight = torch.ones_like(dist_sq, dtype=torch.float32)
        return torch.where(sphere, weight.to(dtype=torch.float32), torch.zeros_like(weight, dtype=torch.float32))
