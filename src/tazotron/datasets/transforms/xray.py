"""Transforms for rendering DRRs (X-rays) from CT volumes via diffdrr."""

from __future__ import annotations

import torch
import torchio as tio
from diffdrr.drr import DRR
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torchvision.transforms import v2

POSE_RANK = 2
POSE_VEC_DIM = 3
EMPTY_EPS = 1e-6


class RenderDRRConfig(BaseModel):
    """Validated configuration for DRR rendering."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sdd: float = 1020.0
    height: int = 200
    delx: float = 2.0
    width: int | None = None
    dely: float | None = None
    renderer: str = "siddon"
    patch_size: int | None = None
    device: torch.device | str | None = None
    reverse_x_axis: bool = True

    def resolved_device(self) -> torch.device | None:
        """Convert an optional device spec into a torch.device."""
        if self.device is None:
            return None
        return self.device if isinstance(self.device, torch.device) else torch.device(self.device)


class RenderDRR(v2.Transform):
    """Renders DRRs from a CT `torchio.Subject` using diffdrr."""

    def __init__(self, config: RenderDRRConfig | dict[str, object] | None = None) -> None:
        super().__init__()
        validated = RenderDRRConfig.model_validate(config or {})
        self.config = validated

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Render DRR for the provided subject."""
        if not isinstance(subject, tio.Subject):
            msg = "RenderDRR expects a torchio.Subject."
            raise TypeError(msg)

        if "volume" not in subject or "density" not in subject or "reorient" not in subject:
            msg = "Subject must be built with diffdrr.read."
            raise KeyError(msg)

        rotations = subject.get("rotations")
        translations = subject.get("translations")
        self._validate_pose(rotations, translations)

        drr_module = DRR(
            subject,
            sdd=self.config.sdd,
            height=self.config.height,
            delx=self.config.delx,
            width=self.config.width,
            dely=self.config.dely,
            reverse_x_axis=self.config.reverse_x_axis,
            patch_size=self.config.patch_size,
            renderer=self.config.renderer,
        )

        target_device = self.config.resolved_device() or rotations.device

        rotations = rotations.to(target_device)
        translations = translations.to(target_device)
        drr_module = drr_module.to(target_device)

        drr_image = drr_module(
            rotations,
            translations,
            parameterization="euler_angles",
            convention="ZXY",
        )

        if (drr_image.max() - drr_image.min()) < EMPTY_EPS:
            msg = "Rendered DRR is empty; camera pose likely misses the volume."
            raise ValueError(msg)

        subject["xray"] = drr_image

        return subject

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
