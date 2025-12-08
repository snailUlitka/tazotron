"""Transforms for rendering DRRs (X-rays) from CT volumes via diffdrr."""

from __future__ import annotations

# ruff: noqa: PLR0913
import copy
from typing import Any

import torch
import torchio as tio
from diffdrr.data import Subject as DiffDRRSubject
from diffdrr.data import read as diffdrr_read
from diffdrr.drr import DRR
from torch import Tensor
from torchvision.transforms import v2

POSE_RANK = 2
POSE_VEC_DIM = 3
DRR_RANK = 4
EMPTY_EPS = 1e-6
TRIM_FRACTION_THRESHOLD = 0.01
TRIM_VAR_THRESHOLD = 1e-4
MIN_TRIM_SIZE = 32
CORNER_FRACTION = 0.05
CONTENT_FRACTION_THRESHOLD = 0.05


class RenderDRR(v2.Transform):
    """Renders DRRs from a CT `torchio.Subject` using diffdrr."""

    def __init__(
        self,
        sdd: float = 1020.0,
        height: int = 512,
        delx: float = 0.5,
        width: int | None = None,
        dely: float | None = None,
        renderer: str = "siddon",
        parameterization: str = "euler_angles",
        convention: str = "ZXY",
        patch_size: int | None = None,
        drr_key: str = "drr",
        rotations_key: str = "rotations",
        translations_key: str = "translations",
        device: torch.device | str | None = None,
        *,
        reverse_x_axis: bool = True,
        inplace: bool = False,
        detach_output: bool = False,
        ct_key: str = "volume",
        label_key: str = "label",
        to_intensity: bool = False,
        normalize: bool = True,
        normalize_quantiles: tuple[float, float] = (0.01, 0.99),
        invert_output: bool = True,
        center_to_label: bool = False,
        center_volume: bool | None = False,
        trim_background: bool = False,
        trim_tolerance: float = 0.1,
        background_value: float | None = None,
    ) -> None:
        super().__init__()
        self.sdd = sdd
        self.height = height
        self.delx = delx
        self.width = width
        self.dely = dely
        self.renderer = renderer
        self.reverse_x_axis = reverse_x_axis
        self.parameterization = parameterization
        self.convention = convention
        self.patch_size = patch_size
        self.drr_key = drr_key
        self.rotations_key = rotations_key
        self.translations_key = translations_key
        self.device = torch.device(device) if device is not None else None
        self.inplace = inplace
        self.detach_output = detach_output
        self.ct_key = ct_key
        self.label_key = label_key
        self.to_intensity = to_intensity
        self.normalize = normalize
        self.normalize_quantiles = normalize_quantiles
        self.invert_output = invert_output
        self.center_to_label = center_to_label
        # If explicitly provided, override diffdrr centering; otherwise center by label when requested.
        self.center_volume = center_volume
        self.trim_background = trim_background
        self.trim_tolerance = trim_tolerance
        self.background_value = background_value

    def __call__(self, sample: Any) -> Any:
        """Render DRR for the provided sample."""
        subject, rotations, translations, container = self._unwrap_sample(sample)
        subject = self._ensure_diffdrr_subject(subject)
        drr_module = self._build_drr(subject)
        target_device = self.device or rotations.device
        rotations = rotations.to(target_device)
        translations = translations.to(target_device)
        drr_module = drr_module.to(target_device)
        drr_image = drr_module(
            rotations,
            translations,
            parameterization=self.parameterization,
            convention=self.convention,
        )
        raw_min = drr_image.min()
        raw_max = drr_image.max()
        if (raw_max - raw_min) < EMPTY_EPS:
            msg = "Rendered DRR is empty; camera pose likely misses the volume. Adjust translation/rotation."
            raise ValueError(msg)
        if self.to_intensity:
            drr_image = torch.exp(-drr_image)
        if self.normalize:
            q_low, q_high = self.normalize_quantiles
            flat = drr_image.flatten(-2)
            drr_min = torch.quantile(
                flat,
                torch.tensor(q_low, device=drr_image.device),
                dim=-1,
                keepdim=True,
            )
            drr_max = torch.quantile(
                flat,
                torch.tensor(q_high, device=drr_image.device),
                dim=-1,
                keepdim=True,
            )
            drr_min = drr_min.unsqueeze(-1)
            drr_max = drr_max.unsqueeze(-1)
            drr_image = (drr_image - drr_min) / (drr_max - drr_min + 1e-8)
            drr_image = drr_image.clamp(0, 1)
        if self.invert_output:
            drr_image = 1 - drr_image
        if self.detach_output:
            drr_image = drr_image.detach()
        if self.trim_background:
            drr_image = self._trim_background(drr_image)
        return self._attach_output(container, drr_image)

    def _unwrap_sample(self, sample: Any) -> tuple[tio.Subject, Tensor, Tensor, Any]:
        if isinstance(sample, tio.Subject):
            subject = sample if self.inplace else copy.deepcopy(sample)
            rotations = subject.get(self.rotations_key)
            translations = subject.get(self.translations_key)
            container = subject
        elif isinstance(sample, dict):
            container = sample if self.inplace else copy.deepcopy(sample)
            subject = container["subject"]
            rotations = container.get(self.rotations_key)
            translations = container.get(self.translations_key)
        else:
            msg = "RenderDRR expects a torchio.Subject or a mapping with a 'subject' key."
            raise TypeError(msg)
        self._validate_pose(rotations, translations)
        return subject, rotations, translations, container

    def _attach_output(self, container: Any, drr_image: Tensor) -> Any:
        if isinstance(container, dict):
            container[self.drr_key] = drr_image
            return container
        container[self.drr_key] = drr_image
        return container

    def _build_drr(self, subject: tio.Subject) -> DRR:
        drr = DRR(
            subject,
            sdd=self.sdd,
            height=self.height,
            delx=self.delx,
            width=self.width,
            dely=self.dely,
            reverse_x_axis=self.reverse_x_axis,
            patch_size=self.patch_size,
            renderer=self.renderer,
        )
        if self.device is not None:
            drr = drr.to(self.device)
        return drr

    def _ensure_diffdrr_subject(self, subject: tio.Subject) -> DiffDRRSubject:
        center_volume = self._resolve_centering(subject)
        data = dict(subject)
        if "volume" not in data:
            if self.ct_key not in data:
                msg = f"Subject missing CT key '{self.ct_key}' required for DRR."
                raise KeyError(msg)
            data["volume"] = data[self.ct_key]
        volume = data["volume"]
        mask = data.get("mask")
        if mask is None:
            for value in data.values():
                if isinstance(value, tio.LabelMap):
                    mask = value
                    break
        subject = diffdrr_read(volume=volume, labelmap=mask, orientation="AP", center_volume=center_volume)
        self._copy_pose(data, subject)
        return subject

    def _resolve_centering(self, subject: tio.Subject) -> bool:
        center_volume = self.center_volume
        if self.center_to_label:
            self._recenter_on_label(subject)
            if center_volume is None:
                center_volume = False
        if center_volume is None:
            center_volume = True
        return center_volume

    def _copy_pose(self, data: dict[str, Any], subject: DiffDRRSubject) -> None:
        for key in ("rotations", "translations"):
            if key in data:
                subject[key] = data[key]

    def _recenter_on_label(self, subject: tio.Subject) -> None:
        """Shift affines so that the label center becomes the isocenter."""
        label_image = self._get_label_image(subject)
        if label_image is None:
            return
        offset = self._compute_center_offset(label_image)
        if offset is None:
            return
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, 3] = offset
        transform_np = transform.numpy()
        for image in subject.get_images(intensity_only=False):
            image.affine = transform_np @ image.affine

    def _get_label_image(self, subject: tio.Subject) -> tio.Image | None:
        if self.label_key in subject and isinstance(subject[self.label_key], tio.LabelMap):
            return subject[self.label_key]
        for image in subject.get_images(intensity_only=False):
            if isinstance(image, tio.LabelMap):
                return image
        return None

    def _compute_center_offset(self, label_image: tio.LabelMap) -> torch.Tensor | None:
        mask = label_image.data > 0
        if not mask.any():
            return None
        coords = mask.nonzero(as_tuple=False)[:, 1:].to(torch.float32)
        center_vox = coords.mean(dim=0)
        affine = torch.as_tensor(label_image.affine, dtype=torch.float32, device=center_vox.device)
        hom = torch.cat([center_vox, torch.tensor([1.0], device=center_vox.device)])
        center_world = (affine @ hom)[:3]
        return -center_world

    def _trim_background(self, drr: Tensor) -> Tensor:
        """Crop uniform background margins from DRR."""
        if drr.dim() != DRR_RANK:
            return drr
        tol = self.trim_tolerance
        mask = drr < (1.0 - tol)
        if not mask.any():
            bg = self._estimate_background(drr)
            mask = drr < (bg - tol)
        if not mask.any():
            return drr
        row_fraction = mask.float().mean(dim=-1).squeeze(1)
        col_fraction = mask.float().mean(dim=-2).squeeze(1)
        row_median = drr.median(dim=-1).values.squeeze(1)
        col_median = drr.median(dim=-2).values.squeeze(1)
        row_keep = (row_fraction > CONTENT_FRACTION_THRESHOLD) | (row_median < (1.0 - tol))
        col_keep = (col_fraction > CONTENT_FRACTION_THRESHOLD) | (col_median < (1.0 - tol))
        if not row_keep.any() or not col_keep.any():
            return drr
        y_indices = row_keep.nonzero(as_tuple=False)[:, -1]
        x_indices = col_keep.nonzero(as_tuple=False)[:, -1]
        y_min, y_max = int(y_indices.min()), int(y_indices.max())
        x_min, x_max = int(x_indices.min()), int(x_indices.max())
        if (y_max - y_min + 1) < MIN_TRIM_SIZE or (x_max - x_min + 1) < MIN_TRIM_SIZE:
            return drr
        return drr[:, :, y_min : y_max + 1, x_min : x_max + 1]

    def _estimate_background(self, drr: Tensor) -> Tensor:
        if self.background_value is not None:
            return torch.tensor(self.background_value, device=drr.device, dtype=drr.dtype)
        h = drr.shape[-2]
        w = drr.shape[-1]
        dh = max(1, int(h * CORNER_FRACTION))
        dw = max(1, int(w * CORNER_FRACTION))
        corners = [
            drr[..., :dh, :dw],
            drr[..., :dh, -dw:],
            drr[..., -dh:, :dw],
            drr[..., -dh:, -dw:],
        ]
        border = torch.cat([c.flatten(-2) for c in corners], dim=-1)
        bg = border.median(dim=-1, keepdim=True).values.unsqueeze(-1)
        return bg

    def _validate_pose(self, rotations: Tensor | None, translations: Tensor | None) -> None:
        if rotations is None or translations is None:
            msg = "Both rotations and translations must be provided for DRR rendering."
            raise ValueError(msg)
        if rotations.dim() != POSE_RANK or rotations.size(-1) != POSE_VEC_DIM:
            msg = f"rotations must have shape (B, {POSE_VEC_DIM})."
            raise ValueError(msg)
        if translations.dim() != POSE_RANK or translations.size(-1) != POSE_VEC_DIM:
            msg = f"translations must have shape (B, {POSE_VEC_DIM})."
            raise ValueError(msg)
        if rotations.shape[0] != translations.shape[0]:
            msg = "rotations and translations must have matching batch sizes."
            raise ValueError(msg)
