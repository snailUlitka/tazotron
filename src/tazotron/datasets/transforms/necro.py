"""Data transformations for volumetric medical images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from diffdrr.data import transform_hu_to_density
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import ndimage
from torch import Tensor
from torchvision.transforms import v2

from tazotron.datasets.ct import COMBINED_FEMORAL_HEAD

if TYPE_CHECKING:
    import torchio as tio

CT_EXPECTED_DIMS = 4
LABEL_EXPECTED_DIMS = 3
LABEL_CHANNELS = 1
LEFT_LABEL = 1
RIGHT_LABEL = 2
LABEL_MODES = ("left", "right", "both")
NECROSIS_HU = 50.0
WORLD_AXES = 3
QUANTILE_EPS = 1e-6


@dataclass(frozen=True)
class _SeverityPreset:
    half_angle_deg: float
    sector_depth_ratio: float
    shell_depth_ratio: float
    low_blobs: int
    high_blobs: int
    low_max_blend: float
    high_max_blend: float
    collapse_max_blend: float
    sigma_range_ratio: tuple[float, float]


@dataclass(frozen=True)
class _HeadGeometry:
    bbox: tuple[slice, slice, slice]
    head_mask_local: Tensor
    world_local: Tensor
    world_points_head: Tensor
    center_world: Tensor
    radius_mm: float
    depth_mm: Tensor
    spacing_mm: tuple[float, float, float]


SEVERITY_PRESETS: dict[str, _SeverityPreset] = {
    "mild": _SeverityPreset(
        half_angle_deg=28.0,
        sector_depth_ratio=0.22,
        shell_depth_ratio=0.08,
        low_blobs=2,
        high_blobs=1,
        low_max_blend=0.30,
        high_max_blend=0.18,
        collapse_max_blend=0.35,
        sigma_range_ratio=(0.12, 0.18),
    ),
    "moderate": _SeverityPreset(
        half_angle_deg=36.0,
        sector_depth_ratio=0.30,
        shell_depth_ratio=0.11,
        low_blobs=3,
        high_blobs=2,
        low_max_blend=0.45,
        high_max_blend=0.25,
        collapse_max_blend=0.55,
        sigma_range_ratio=(0.14, 0.22),
    ),
    "severe": _SeverityPreset(
        half_angle_deg=44.0,
        sector_depth_ratio=0.38,
        shell_depth_ratio=0.14,
        low_blobs=4,
        high_blobs=3,
        low_max_blend=0.60,
        high_max_blend=0.35,
        collapse_max_blend=0.75,
        sigma_range_ratio=(0.18, 0.28),
    ),
}


class AddLateAVNLikeNecrosisV1Config(BaseModel):
    """Validated configuration for the late AVN-like femoral-head transform."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    target_head: Literal["left", "right", "random"] = "random"
    severity: Literal["mild", "moderate", "severe"] = "moderate"
    bone_attenuation_multiplier: float = 1.0
    label_name: str = COMBINED_FEMORAL_HEAD
    left_id: int = LEFT_LABEL
    right_id: int = RIGHT_LABEL
    seed: int | None = None
    generator: torch.Generator | None = None
    min_head_voxels: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def _validate_generator_settings(self) -> AddLateAVNLikeNecrosisV1Config:
        if self.seed is not None and self.generator is not None:
            msg = "seed and generator cannot be set at the same time."
            raise ValueError(msg)
        return self


class AddLateAVNLikeNecrosisV1(v2.Transform):
    """Apply a sector-based late AVN-like edit to exactly one femoral head.

    This V1 transform is intended for CT volumes that already use an anatomical
    RAS world orientation in the affine. It models a late AVN-like pattern via
    local intensity remodeling and pseudo-collapse in a superior-lateral
    subchondral sector, without any spatial warp. The edit is localized to one
    femoral head and is suitable for downstream DRR rendering, not for
    medically exhaustive staging.

    Required subject fields
    -----------------------
    - ``subject["volume"]``: CT tensor shaped ``(1, D, H, W)``
    - ``subject[label_name]``: combined femoral-head labels shaped
      ``(D, H, W)`` or ``(1, D, H, W)``

    Optional subject field
    ----------------------
    - ``subject["density"]``: recomputed from the updated CT when present

    Notes
    -----
    The affine is interpreted in RAS terms: ``+X = Right``, ``+Y = Anterior``,
    ``+Z = Superior``. The superior-lateral pole therefore uses ``-X`` for the
    left head and ``+X`` for the right head, with a small AP jitter.
    """

    def __init__(self, config: AddLateAVNLikeNecrosisV1Config | dict[str, object] | None = None) -> None:
        super().__init__()
        self.config = AddLateAVNLikeNecrosisV1Config.model_validate(config or {})
        self._seeded_generator: torch.Generator | None = None
        self._seeded_generator_device: torch.device | None = None

    def __call__(self, subject: Any) -> Any:  # noqa: PLR0911
        """Apply the transform to a ``torchio.Subject`` in place."""
        self._validate_subject(subject)
        volume_image = subject["volume"]
        label_image = subject[self.config.label_name]
        volume = _image_to_tensor(volume_image)
        label = _label_to_3d(_image_to_tensor(label_image))

        if volume.ndim != CT_EXPECTED_DIMS:
            msg = "Expected CT tensor shaped (1, D, H, W)."
            raise ValueError(msg)
        if self.config.probability <= 0.0:
            return subject

        generator = self._get_generator(volume.device)
        if self.config.probability < 1.0 and self._sample_uniform(volume.device, generator) > self.config.probability:
            return subject

        head_name = self._select_target_head(label, volume.device, generator)
        if head_name is None:
            return subject

        head_mask = self._head_mask(label, head_name)
        if int(head_mask.sum().item()) < self.config.min_head_voxels:
            return subject

        geometry = self._compute_head_geometry(head_mask, label_image.affine, volume.device)
        if geometry is None:
            return subject

        preset = SEVERITY_PRESETS[self.config.severity]
        sector_weight, shell_weight = self._build_sector_maps(
            geometry=geometry,
            head_name=head_name,
            preset=preset,
            device=volume.device,
            generator=generator,
        )
        if float(sector_weight.max().item()) <= QUANTILE_EPS:
            return subject

        updated_volume = self._apply_late_avn_pattern(
            volume=volume,
            geometry=geometry,
            sector_weight=sector_weight,
            shell_weight=shell_weight,
            preset=preset,
            generator=generator,
        )
        if torch.equal(updated_volume, volume):
            return subject

        _assign_image_tensor(volume_image, updated_volume)
        if "density" in subject:
            self._update_density(subject, updated_volume, volume_image)
        return subject

    def _validate_subject(self, subject: Any) -> None:
        if not hasattr(subject, "__contains__") or not hasattr(subject, "__getitem__"):
            msg = "AddLateAVNLikeNecrosisV1 expects a torchio.Subject."
            raise TypeError(msg)
        if "volume" not in subject:
            msg = "Subject must contain 'volume'."
            raise KeyError(msg)
        if self.config.label_name not in subject:
            msg = f"Subject must contain '{self.config.label_name}'."
            raise KeyError(msg)

    def _select_target_head(
        self,
        label: Tensor,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> str | None:
        available: list[str] = []
        if torch.any(label == self.config.left_id):
            available.append("left")
        if torch.any(label == self.config.right_id):
            available.append("right")
        if not available:
            return None
        if self.config.target_head != "random":
            return self.config.target_head if self.config.target_head in available else None
        if len(available) == 1:
            return available[0]
        selected_index = int(
            torch.randint(len(available), (1,), generator=generator, device=device, dtype=torch.int64).item(),
        )
        return available[selected_index]

    def _head_mask(self, label: Tensor, head_name: str) -> Tensor:
        label_id = self.config.left_id if head_name == "left" else self.config.right_id
        return label == label_id

    def _compute_head_geometry(
        self,
        head_mask: Tensor,
        affine: Any,
        device: torch.device,
    ) -> _HeadGeometry | None:
        coords = head_mask.nonzero(as_tuple=False)
        if coords.numel() == 0:
            return None

        margin = 1
        mins = torch.clamp(coords.min(dim=0).values - margin, min=0)
        maxs = torch.minimum(
            coords.max(dim=0).values + margin,
            torch.as_tensor(head_mask.shape, device=device) - 1,
        )
        bbox = tuple(slice(int(mins[idx].item()), int(maxs[idx].item()) + 1) for idx in range(WORLD_AXES))
        head_mask_local = head_mask[bbox]

        affine_tensor = torch.as_tensor(affine, dtype=torch.float32, device=device)
        spacing_mm = tuple(
            float(torch.linalg.norm(affine_tensor[:WORLD_AXES, axis]).item())
            for axis in range(WORLD_AXES)
        )
        world_points_head = self._indices_to_world(coords.to(torch.float32), affine_tensor)
        center_world = world_points_head.mean(dim=0)
        distances = torch.linalg.norm(world_points_head - center_world, dim=1)
        radius_mm = float(torch.quantile(distances, 0.95).item())
        if radius_mm <= QUANTILE_EPS:
            return None

        depth_mm_np = ndimage.distance_transform_edt(head_mask_local.cpu().numpy(), sampling=spacing_mm)
        depth_mm = torch.as_tensor(depth_mm_np, dtype=torch.float32, device=device)
        world_local = self._world_grid_for_bbox(bbox, affine_tensor, device)
        return _HeadGeometry(
            bbox=bbox,
            head_mask_local=head_mask_local,
            world_local=world_local,
            world_points_head=world_points_head,
            center_world=center_world,
            radius_mm=radius_mm,
            depth_mm=depth_mm,
            spacing_mm=spacing_mm,
        )

    def _build_sector_maps(
        self,
        *,
        geometry: _HeadGeometry,
        head_name: str,
        preset: _SeverityPreset,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        lateral_sign = -1.0 if head_name == "left" else 1.0
        ap_jitter = self._sample_uniform_range(-0.15, 0.15, device, generator)
        pole = torch.tensor(
            [lateral_sign, ap_jitter, 1.0],
            dtype=torch.float32,
            device=device,
        )
        pole = pole / torch.linalg.norm(pole)

        radial = geometry.world_local - geometry.center_world
        radial_norm = torch.linalg.norm(radial, dim=-1, keepdim=True).clamp_min(QUANTILE_EPS)
        radial_dir = radial / radial_norm
        dot = torch.sum(radial_dir * pole, dim=-1)

        outer_angle_deg = preset.half_angle_deg + 8.0
        angular_weight = _smoothstep(
            torch.cos(torch.deg2rad(torch.tensor(outer_angle_deg, device=device))),
            torch.cos(torch.deg2rad(torch.tensor(preset.half_angle_deg, device=device))),
            dot,
        )

        sector_depth_mm = preset.sector_depth_ratio * geometry.radius_mm
        shell_depth_mm = preset.shell_depth_ratio * geometry.radius_mm
        sector_depth_weight = 1.0 - _smoothstep(
            torch.tensor(sector_depth_mm * 0.6, dtype=torch.float32, device=device),
            torch.tensor(sector_depth_mm, dtype=torch.float32, device=device),
            geometry.depth_mm,
        )
        shell_weight = 1.0 - _smoothstep(
            torch.tensor(shell_depth_mm * 0.55, dtype=torch.float32, device=device),
            torch.tensor(shell_depth_mm, dtype=torch.float32, device=device),
            geometry.depth_mm,
        )
        sector_weight = angular_weight * sector_depth_weight * geometry.head_mask_local.to(torch.float32)
        shell_weight = shell_weight * angular_weight * geometry.head_mask_local.to(torch.float32)
        return sector_weight, shell_weight

    def _apply_late_avn_pattern(  # noqa: PLR0913
        self,
        *,
        volume: Tensor,
        geometry: _HeadGeometry,
        sector_weight: Tensor,
        shell_weight: Tensor,
        preset: _SeverityPreset,
        generator: torch.Generator | None,
    ) -> Tensor:
        bbox = geometry.bbox
        local_ct = volume[(0, *bbox)].clone()
        head_values = local_ct[geometry.head_mask_local]
        if head_values.numel() == 0:
            return volume

        targets = self._estimate_targets(head_values)
        low_field = self._blob_field(
            world_local=geometry.world_local,
            candidate_mask=sector_weight > QUANTILE_EPS,
            num_blobs=preset.low_blobs,
            sigma_range_ratio=preset.sigma_range_ratio,
            radius_mm=geometry.radius_mm,
            generator=generator,
        )
        high_field = self._blob_field(
            world_local=geometry.world_local,
            candidate_mask=sector_weight > QUANTILE_EPS,
            num_blobs=preset.high_blobs,
            sigma_range_ratio=preset.sigma_range_ratio,
            radius_mm=geometry.radius_mm,
            generator=generator,
        )
        irregularity = self._blob_field(
            world_local=geometry.world_local,
            candidate_mask=shell_weight > QUANTILE_EPS,
            num_blobs=int(torch.randint(1, 3, (1,), generator=generator, device=volume.device).item()),
            sigma_range_ratio=(0.24, 0.38),
            radius_mm=geometry.radius_mm,
            generator=generator,
        )

        low_weight = sector_weight * low_field * preset.low_max_blend
        high_weight = sector_weight * high_field * preset.high_max_blend
        high_weight = high_weight * (1.0 - low_weight).clamp_min(0.0)
        updated_local = local_ct + low_weight * (targets["low"] - local_ct) + high_weight * (targets["high"] - local_ct)

        cap_weight = shell_weight * irregularity * preset.collapse_max_blend
        updated_local = updated_local + cap_weight * (targets["collapse"] - updated_local)

        clamp_min = torch.tensor(targets["clamp_min"], dtype=updated_local.dtype, device=updated_local.device)
        clamp_max = torch.tensor(targets["clamp_max"], dtype=updated_local.dtype, device=updated_local.device)
        updated_local[geometry.head_mask_local] = torch.clamp(
            updated_local[geometry.head_mask_local],
            clamp_min,
            clamp_max,
        )

        updated_volume = volume.clone()
        updated_volume[(0, *bbox)] = torch.where(geometry.head_mask_local, updated_local, updated_volume[(0, *bbox)])
        return updated_volume

    def _estimate_targets(self, head_values: Tensor) -> dict[str, float]:
        quantiles = torch.quantile(
            head_values.to(torch.float32),
            torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], device=head_values.device),
        )
        q05, q10, q25, q50, q75, q90, q95 = (float(value.item()) for value in quantiles)
        severity = self.config.severity

        low_target = {"mild": q25, "moderate": 0.5 * (q10 + q25), "severe": q10}[severity]
        high_target = {"mild": q75, "moderate": 0.5 * (q75 + q90), "severe": q90}[severity]
        collapse_scale = {"mild": 0.10, "moderate": 0.20, "severe": 0.30}[severity]
        collapse_target = max(q05, q10 - collapse_scale * (q50 - q10))
        return {
            "low": low_target,
            "high": high_target,
            "collapse": collapse_target,
            "clamp_min": q05,
            "clamp_max": q95,
        }

    def _blob_field(  # noqa: PLR0913
        self,
        *,
        world_local: Tensor,
        candidate_mask: Tensor,
        num_blobs: int,
        sigma_range_ratio: tuple[float, float],
        radius_mm: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        field = torch.zeros(candidate_mask.shape, dtype=torch.float32, device=candidate_mask.device)
        coords = world_local[candidate_mask]
        if coords.numel() == 0 or num_blobs <= 0:
            return field

        for _ in range(num_blobs):
            center_index = int(torch.randint(coords.shape[0], (1,), generator=generator, device=coords.device).item())
            sigma_ratio = self._sample_uniform_range(
                sigma_range_ratio[0],
                sigma_range_ratio[1],
                coords.device,
                generator,
            )
            sigma_mm = max(radius_mm * sigma_ratio, QUANTILE_EPS)
            center = coords[center_index]
            squared_dist = torch.sum((world_local - center) ** 2, dim=-1)
            field = field + torch.exp(-0.5 * squared_dist / (sigma_mm**2))

        masked_max = float(field[candidate_mask].max().item()) if torch.any(candidate_mask) else 0.0
        if masked_max <= QUANTILE_EPS:
            return torch.zeros_like(field)
        field = field / masked_max
        return field * candidate_mask.to(torch.float32)

    def _indices_to_world(self, coords: Tensor, affine: Tensor) -> Tensor:
        ones = torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
        homogeneous = torch.cat((coords, ones), dim=1)
        world = homogeneous @ affine.mT
        return world[:, :WORLD_AXES]

    def _world_grid_for_bbox(self, bbox: tuple[slice, slice, slice], affine: Tensor, device: torch.device) -> Tensor:
        axis_0 = torch.arange(bbox[0].start, bbox[0].stop, dtype=torch.float32, device=device)
        axis_1 = torch.arange(bbox[1].start, bbox[1].stop, dtype=torch.float32, device=device)
        axis_2 = torch.arange(bbox[2].start, bbox[2].stop, dtype=torch.float32, device=device)
        grid_0, grid_1, grid_2 = torch.meshgrid(axis_0, axis_1, axis_2, indexing="ij")
        coords = torch.stack((grid_0, grid_1, grid_2), dim=-1)
        flat = coords.reshape(-1, WORLD_AXES)
        flat_world = self._indices_to_world(flat, affine)
        return flat_world.reshape(*coords.shape[:-1], WORLD_AXES)

    def _update_density(self, subject: Any, updated_volume: Tensor, volume_image: Any) -> None:
        density_image = subject["density"]
        if density_image is volume_image:
            return
        updated_density = transform_hu_to_density(updated_volume, self.config.bone_attenuation_multiplier)
        _assign_image_tensor(density_image, updated_density)

    def _sample_uniform(self, device: torch.device, generator: torch.Generator | None) -> float:
        return float(torch.rand((), device=device, generator=generator).item())

    def _sample_uniform_range(
        self,
        low: float,
        high: float,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> float:
        return low + (high - low) * self._sample_uniform(device, generator)

    def _get_generator(self, device: torch.device) -> torch.Generator | None:
        if self.config.generator is not None:
            generator_device = getattr(self.config.generator, "device", device)
            if generator_device != device:
                msg = (
                    "Provided generator device does not match the tensor device. "
                    "Create the generator on the same device as the tensors."
                )
                raise ValueError(msg)
            return self.config.generator
        if self.config.seed is None:
            return None
        if self._seeded_generator is None or self._seeded_generator_device != device:
            seeded = torch.Generator(device=device)
            seeded.manual_seed(self.config.seed)
            self._seeded_generator = seeded
            self._seeded_generator_device = device
        return self._seeded_generator


class AddRandomNecrosis(v2.Transform):
    """Inject necrosis into a femoral head mask by setting random voxels to a fixed HU value.

    This transform is tailored for subjects produced by `tazotron.datasets.ct.CTDataset`:
    it expects `subject["volume"]` and `subject["label_combined_femoral_head"]`, and applies changes in place.

    Parameters
    ----------
    intensity:
        Fraction of voxels inside the target mask to replace with necrosis. Must be in [0, 1].
    label_mode:
        Which femoral head label(s) to use for the necrosis mask: "left", "right", or "both".
    bone_attenuation_multiplier:
        Multiplier for high-density voxels when recomputing the density image.
    seed:
        Optional base seed for deterministic voxel selection across calls.
    generator:
        Optional torch.Generator to drive voxel selection. Cannot be set together with seed.
    """

    def __init__(
        self,
        intensity: float = 0.1,
        *,
        label_mode: str = "both",
        bone_attenuation_multiplier: float = 1.0,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 <= intensity <= 1.0):
            msg = "intensity must be in the range [0, 1]."
            raise ValueError(msg)
        if seed is not None and generator is not None:
            msg = "seed and generator cannot be set at the same time."
            raise ValueError(msg)
        self.intensity = intensity
        self.label_mode = self._normalize_label_mode(label_mode)
        self.bone_attenuation_multiplier = bone_attenuation_multiplier
        self.seed = seed
        self._user_generator = generator
        self._seeded_generator: torch.Generator | None = None
        self._seeded_generator_device: torch.device | None = None

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Apply the transform to a `torchio.Subject` in place."""
        volume_image = subject["volume"]
        label_image = subject[COMBINED_FEMORAL_HEAD]

        volume = _image_to_tensor(volume_image)
        label = _image_to_tensor(label_image)
        mask = self._mask_from_label(label)
        if not mask.any() or self.intensity == 0.0:
            return subject

        updated_volume = self._apply_necrosis(volume, mask)
        _assign_image_tensor(volume_image, updated_volume)
        if "density" in subject:
            density_image = subject["density"]
            if density_image is not volume_image:
                updated_density = transform_hu_to_density(
                    updated_volume,
                    self.bone_attenuation_multiplier,
                )
                _assign_image_tensor(density_image, updated_density)
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

        generator = self._get_generator(mask.device)

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
        label_3d = _label_to_3d(label)
        if self.label_mode == "left":
            return label_3d == LEFT_LABEL
        if self.label_mode == "right":
            return label_3d == RIGHT_LABEL
        return (label_3d == LEFT_LABEL) | (label_3d == RIGHT_LABEL)

    def _normalize_label_mode(self, label_mode: str) -> str:
        if label_mode not in LABEL_MODES:
            msg = f"label_mode must be one of {LABEL_MODES}."
            raise ValueError(msg)
        return label_mode

    def _get_generator(self, device: torch.device) -> torch.Generator | None:
        if self._user_generator is not None:
            generator_device = getattr(self._user_generator, "device", device)
            if generator_device != device:
                msg = (
                    "Provided generator device does not match the mask device. "
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


def _label_to_3d(label: Tensor) -> Tensor:
    if label.ndim == CT_EXPECTED_DIMS and label.shape[0] == LABEL_CHANNELS:
        return label[0]
    if label.ndim != LABEL_EXPECTED_DIMS:
        msg = "Expected labelmap shaped (D, H, W) or (1, D, H, W)."
        raise ValueError(msg)
    return label


def _image_to_tensor(image: tio.Image) -> Tensor:
    if hasattr(image, "data"):
        return image.data
    msg = "Failed to extract tensor from torchio.Image."
    raise TypeError(msg)


def _assign_image_tensor(image: tio.Image, tensor: Tensor) -> None:
    if hasattr(image, "set_data"):
        image.set_data(tensor)
        return
    msg = "Failed to assign tensor to torchio.Image."
    raise TypeError(msg)


def _smoothstep(edge0: Tensor, edge1: Tensor, value: Tensor) -> Tensor:
    denominator = torch.clamp(edge1 - edge0, min=QUANTILE_EPS)
    scaled = torch.clamp((value - edge0) / denominator, min=0.0, max=1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)
