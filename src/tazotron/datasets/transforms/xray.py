"""Transforms for rendering DRRs (X-rays) from CT volumes via diffdrr."""

from __future__ import annotations

# ruff: noqa: PLR0913
import copy
from typing import Any

import torch
import torchio as tio
from diffdrr.drr import DRR
from torch import Tensor
from torchvision.transforms import v2

POSE_RANK = 2
POSE_VEC_DIM = 3


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
        compile_renderer: bool = False,
        degrees: bool = False,
        inplace: bool = False,
        detach_output: bool = False,
    ) -> None:
        super().__init__()
        self.sdd = sdd
        self.height = height
        self.delx = delx
        self.width = width
        self.dely = dely
        self.renderer = renderer
        self.reverse_x_axis = reverse_x_axis
        self.compile_renderer = compile_renderer
        self.parameterization = parameterization
        self.convention = convention
        self.degrees = degrees
        self.patch_size = patch_size
        self.drr_key = drr_key
        self.rotations_key = rotations_key
        self.translations_key = translations_key
        self.device = torch.device(device) if device is not None else None
        self.inplace = inplace
        self.detach_output = detach_output

    def __call__(self, sample: Any) -> Any:
        """Render DRR for the provided sample."""
        subject, rotations, translations, container = self._unwrap_sample(sample)
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
        degrees=self.degrees,
    )
        if self.detach_output:
            drr_image = drr_image.detach()
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
            compile_renderer=self.compile_renderer,
        )
        if self.device is not None:
            drr = drr.to(self.device)
        return drr

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
