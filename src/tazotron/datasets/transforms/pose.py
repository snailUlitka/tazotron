"""Pose-selection transforms for DRR rendering."""

from __future__ import annotations

from typing import Any

import torch
import torchio as tio
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torchvision.transforms import v2

from tazotron.datasets.ct import COMBINED_FEMORAL_HEAD

POSE_VEC_DIM = 3


class InvalidBilateralMaskError(ValueError):
    """Raised when bilateral femoral-head framing cannot be computed."""


class AutoBilateralHipPoseConfig(BaseModel):
    """Configuration for bilateral-hip auto-pose selection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label_name: str = COMBINED_FEMORAL_HEAD
    left_id: int = 1
    right_id: int = 2
    sid: float = 800.0
    sdd: float = 1020.0
    width: int = 200
    height: int = 200
    min_x_mm: float = 280.0
    min_z_mm: float = 280.0
    margin_x_mm: float = 20.0
    margin_z_mm: float = 20.0
    reverse_x_axis: bool = True
    renderer: str = "siddon"


class AutoBilateralHipPose(v2.Transform):
    """Select a deterministic AP pose and detector FOV from bilateral femoral-head masks."""

    def __init__(self, config: AutoBilateralHipPoseConfig | dict[str, object] | None = None) -> None:
        super().__init__()
        self.config = AutoBilateralHipPoseConfig.model_validate(config or {})

    def __call__(self, subject: Any) -> Any:
        """Attach pose tensors and per-subject DRR config to the input subject."""
        if not isinstance(subject, tio.Subject):
            msg = "AutoBilateralHipPose expects a torchio.Subject."
            raise TypeError(msg)

        label_image = subject[self.config.label_name]
        left_points = self._label_points_world(label_image, self.config.left_id)
        right_points = self._label_points_world(label_image, self.config.right_id)
        if left_points.numel() == 0 or right_points.numel() == 0:
            msg = "Both left and right femoral-head masks are required for auto pose."
            raise InvalidBilateralMaskError(msg)

        left_center = left_points.mean(dim=0)
        right_center = right_points.mean(dim=0)
        target_center = (left_center + right_center) / 2.0

        union_points = torch.cat((left_points, right_points), dim=0)
        bbox_min = union_points.min(dim=0).values
        bbox_max = union_points.max(dim=0).values
        bbox_x_mm = float((bbox_max[0] - bbox_min[0]).item())
        bbox_z_mm = float((bbox_max[2] - bbox_min[2]).item())
        coverage_x_mm = max(bbox_x_mm + 2.0 * self.config.margin_x_mm, self.config.min_x_mm)
        coverage_z_mm = max(bbox_z_mm + 2.0 * self.config.margin_z_mm, self.config.min_z_mm)

        delx = coverage_x_mm * self.config.sdd / (self.config.sid * self.config.width)
        dely = coverage_z_mm * self.config.sdd / (self.config.sid * self.config.height)

        subject["rotations"] = torch.zeros((1, POSE_VEC_DIM), dtype=torch.float32)
        subject["translations"] = torch.tensor(
            [[target_center[0].item(), target_center[1].item() + self.config.sid, target_center[2].item()]],
            dtype=torch.float32,
        )
        subject["drr_config"] = {
            "sdd": self.config.sdd,
            "width": self.config.width,
            "height": self.config.height,
            "delx": float(delx),
            "dely": float(dely),
            "reverse_x_axis": self.config.reverse_x_axis,
            "renderer": self.config.renderer,
        }
        subject["pose_metadata"] = {
            "mode": "autopose",
            "sid_mm": self.config.sid,
            "sdd_mm": self.config.sdd,
            "bbox_x_mm": bbox_x_mm,
            "bbox_z_mm": bbox_z_mm,
            "coverage_x_mm": coverage_x_mm,
            "coverage_z_mm": coverage_z_mm,
            "target_center_world_x": float(target_center[0].item()),
            "target_center_world_y": float(target_center[1].item()),
            "target_center_world_z": float(target_center[2].item()),
        }
        return subject

    @staticmethod
    def _label_points_world(label_image: tio.Image, label_id: int) -> Tensor:
        label = label_image.data[0]
        indices = (label == label_id).nonzero(as_tuple=False)
        if indices.numel() == 0:
            return torch.empty((0, POSE_VEC_DIM), dtype=torch.float32, device=label.device)

        affine = torch.as_tensor(label_image.affine, dtype=torch.float32, device=indices.device)
        homogenous = torch.cat(
            (
                indices.to(dtype=torch.float32),
                torch.ones((indices.shape[0], 1), dtype=torch.float32, device=indices.device),
            ),
            dim=1,
        )
        world = homogenous @ affine.mT
        return world[:, :POSE_VEC_DIM]
