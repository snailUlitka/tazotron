"""Composite transforms for CT-to-DRR pipelines."""

from __future__ import annotations

from typing import Any

import torch
import torchio as tio
from torchvision.transforms import v2

from tazotron.datasets.transforms.crop import BilateralHipROICrop
from tazotron.datasets.transforms.necrosis import AddRandomNecrosis
from tazotron.datasets.transforms.xray import RenderDRR


class CTToXRTransform(v2.Transform):
    """Pipeline: crop ROI by label and render DRR."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        label_name: str = "label",
        left_id: int = 1,
        right_id: int = 2,
        size_si_mm: float = 260.0,
        min_lr_mm: float = 240.0,
        margin_lr_mm: float = 20.0,
        size_ap_mm: float = 160.0,
        crop_depth: bool = False,
        skip_crop: bool = False,
        render_kwargs: dict[str, Any] | None = None,
        necrosis_spots: int | tuple[int, int] = 0,
        necrosis_drop: tuple[float, float] | None = None,
        autopose: bool = True,
        camera_offset_mm: float = 800.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.skip_crop = skip_crop
        self.autopose = autopose
        self.camera_offset_mm = camera_offset_mm
        self.necrosis = self._build_necrosis(
            necrosis_spots,
            necrosis_drop,
            label_name,
            left_id,
            right_id,
            inplace=inplace,
        )
        self.crop = BilateralHipROICrop(
            label_name=label_name,
            left_id=left_id,
            right_id=right_id,
            size_si_mm=size_si_mm,
            min_lr_mm=min_lr_mm,
            margin_lr_mm=margin_lr_mm,
            size_ap_mm=size_ap_mm,
            crop_depth=crop_depth,
            inplace=inplace,
        )
        default_render = {
            "sdd": 1020.0,
            "height": 512,
            "width": 512,
            "delx": 0.6,
            "dely": 0.6,
            "renderer": "siddon",
            "detach_output": True,
            "trim_background": True,
            "center_volume": False,
            "center_to_label": False,
            "invert_output": False,
        }
        render_params = {**default_render, **(render_kwargs or {})}
        if self.autopose:
            render_params.setdefault("center_to_label", False)
            render_params.setdefault("center_volume", False)
        self.render = RenderDRR(label_key=label_name, **render_params)

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Apply crop then render DRR."""
        working = self.necrosis(subject) if self.necrosis is not None else subject
        cropped = working if self.skip_crop else self.crop(working)
        if self.autopose:
            self._ensure_pose(cropped)
        return self.render(cropped)

    def _build_necrosis(  # noqa: PLR0913
        self,
        necrosis_spots: int | tuple[int, int],
        necrosis_drop: tuple[float, float] | None,
        label_name: str,
        left_id: int,
        right_id: int,
        *,
        inplace: bool,
    ) -> AddRandomNecrosis | None:
        if isinstance(necrosis_spots, int):
            if necrosis_spots < 0:
                msg = "necrosis_spots cannot be negative."
                raise ValueError(msg)
            if necrosis_spots == 0:
                return None
            spots_range = (necrosis_spots, necrosis_spots)
        else:
            if necrosis_spots[0] < 0 or necrosis_spots[1] < necrosis_spots[0]:
                msg = "necrosis_spots range must be non-negative and ordered."
                raise ValueError(msg)
            if necrosis_spots[1] == 0:
                return None
            spots_range = necrosis_spots
        drop_range = necrosis_drop
        if drop_range is not None and (drop_range[0] < 0 or drop_range[1] < drop_range[0]):
            msg = "necrosis_drop must be ordered and non-negative."
            raise ValueError(msg)
        return AddRandomNecrosis(
            probability=1.0,
            num_spots=spots_range,
            target_labels=(left_id, right_id),
            intensity_drop=drop_range if drop_range is not None else (0.35, 0.65),
            ct_key="volume",
            label_key=label_name,
            inplace=inplace,
        )

    def _ensure_pose(self, subject: tio.Subject) -> None:
        if subject.get("rotations") is None:
            subject["rotations"] = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        if subject.get("translations") is None:
            subject["translations"] = torch.tensor([self._autopose_translation(subject)], dtype=torch.float32)

    def _autopose_translation(self, subject: tio.Subject) -> tuple[float, float, float]:
        label = subject.get(self.render.label_key)
        if label is None or not isinstance(label, tio.LabelMap):
            msg = "Label map required for autopose."
            raise ValueError(msg)
        data = label.data[0]
        left_mask = data == self.crop.left_id
        right_mask = data == self.crop.right_id
        if not left_mask.any() or not right_mask.any():
            msg = "Both left and right labels are required for autopose."
            raise ValueError(msg)
        centers = []
        for mask in (left_mask, right_mask):
            coords = mask.nonzero(as_tuple=False).to(dtype=torch.float32)
            hom = torch.cat([coords, torch.ones_like(coords[:, :1])], dim=1).T
            affine = torch.as_tensor(label.affine, dtype=torch.float32, device=coords.device)
            world = (affine @ hom)[:3]
            centers.append(world.mean(dim=1))
        midpoint = (centers[0] + centers[1]) / 2.0
        center_x, center_y, center_z = midpoint
        translation_y = center_y + self.camera_offset_mm
        return (center_x.item(), translation_y.item(), center_z.item())
