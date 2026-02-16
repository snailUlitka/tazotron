"""Transform to add femoral head masks derived from femur masks."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torchio as tio
from torch import Tensor
from torchvision.transforms import v2

from tazotron.datasets.ct import (
    COMBINED_FEMORAL_HEAD,
    FEMORAL_HEAD_LEFT,
    FEMORAL_HEAD_RIGHT,
    FEMUR_LEFT,
    FEMUR_RIGHT,
)

LABEL_EXPECTED_DIMS = 3
LABEL_CHANNELS = 1
WORLD_SUPERIOR_AXIS = 2
ROUNDNESS_EPSILON = 1e-6
PLANE_REFERENCE_PARALLEL_THRESHOLD = 0.9
MIN_ROUNDNESS_SLICE_POINTS = 3
ELLIPSE_AXES_COUNT = 2
NEIGHBOR_OFFSETS_26 = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
)


class AddFemoralHeadMasks(v2.Transform):
    """Add femoral head labelmaps based on left/right femur masks.

    Expects a `torchio.Subject` that contains femur masks under `femur_left_key` and `femur_right_key`.
    Adds femoral head masks under `femoral_left_key` and `femoral_right_key`. Optionally updates
    the combined femoral head label entry when present.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        femur_left_key: str = FEMUR_LEFT,
        femur_right_key: str = FEMUR_RIGHT,
        femoral_left_key: str = FEMORAL_HEAD_LEFT,
        femoral_right_key: str = FEMORAL_HEAD_RIGHT,
        area_ratio_threshold: float = 0.65,
        min_slices_from_top: int = 3,
        confirm_slices: int = 2,
        roundness_ratio_threshold: float = 1.35,
        roundness_confirm_slices: int = 2,
        roundness_backstep_slices: int = 1,
        min_component_voxels: int = 100,
        update_combined_label: bool = True,
        overwrite: bool = True,
    ) -> None:
        super().__init__()
        if not (0.0 < area_ratio_threshold < 1.0):
            msg = "area_ratio_threshold must be in the range (0, 1)."
            raise ValueError(msg)
        if min_slices_from_top < 1:
            msg = "min_slices_from_top must be a positive integer."
            raise ValueError(msg)
        if confirm_slices < 1:
            msg = "confirm_slices must be a positive integer."
            raise ValueError(msg)
        if roundness_ratio_threshold < 1.0:
            msg = "roundness_ratio_threshold must be >= 1.0."
            raise ValueError(msg)
        if roundness_confirm_slices < 1:
            msg = "roundness_confirm_slices must be a positive integer."
            raise ValueError(msg)
        if roundness_backstep_slices < 0:
            msg = "roundness_backstep_slices must be >= 0."
            raise ValueError(msg)
        if min_component_voxels < 1:
            msg = "min_component_voxels must be a positive integer."
            raise ValueError(msg)
        self.femur_left_key = femur_left_key
        self.femur_right_key = femur_right_key
        self.femoral_left_key = femoral_left_key
        self.femoral_right_key = femoral_right_key
        self.area_ratio_threshold = area_ratio_threshold
        self.min_slices_from_top = min_slices_from_top
        self.confirm_slices = confirm_slices
        self.roundness_ratio_threshold = roundness_ratio_threshold
        self.roundness_confirm_slices = roundness_confirm_slices
        self.roundness_backstep_slices = roundness_backstep_slices
        self.min_component_voxels = min_component_voxels
        self.update_combined_label = update_combined_label
        self.overwrite = overwrite

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Add left/right femoral head masks to a subject."""
        if not isinstance(subject, tio.Subject):
            msg = "AddFemoralHeadMasks expects a torchio.Subject."
            raise TypeError(msg)

        femur_left = subject.get(self.femur_left_key)
        femur_right = subject.get(self.femur_right_key)
        if femur_left is None or femur_right is None:
            msg = "Both femur masks must be present to derive femoral head labels."
            raise ValueError(msg)
        if not isinstance(femur_left, tio.LabelMap) or not isinstance(femur_right, tio.LabelMap):
            msg = "Femur masks must be torchio.LabelMap instances."
            raise TypeError(msg)
        if femur_left.data.shape != femur_right.data.shape:
            msg = "Femur mask shapes must match."
            raise ValueError(msg)

        left_existing = subject.get(self.femoral_left_key)
        right_existing = subject.get(self.femoral_right_key)
        if not self.overwrite and left_existing is not None and right_existing is not None:
            return subject

        femoral_left, femoral_right = self._infer_femoral_head_masks(femur_left, femur_right)
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
        femur_left: tio.LabelMap,
        femur_right: tio.LabelMap,
    ) -> tuple[tio.LabelMap, tio.LabelMap]:
        """Return femoral head masks for left/right femur labelmaps."""
        left_tensor = self._extract_femoral_head_mask(femur_left.data, affine=femur_left.affine)
        right_tensor = self._extract_femoral_head_mask(femur_right.data, affine=femur_right.affine)
        return (
            tio.LabelMap(tensor=left_tensor, affine=femur_left.affine),
            tio.LabelMap(tensor=right_tensor, affine=femur_right.affine),
        )

    def _extract_femoral_head_mask(self, femur_tensor: Tensor, *, affine: np.ndarray) -> Tensor:
        mask, output = self._extract_mask(femur_tensor)
        if not mask.any():
            return output

        component = self._largest_connected_component(mask)
        if int(component.sum().item()) < self.min_component_voxels:
            return output

        coords = component.nonzero(as_tuple=False).to(torch.float64)
        axis = self._principal_axis(coords)
        projections = coords @ axis
        superior = self._superior_coordinates(coords, affine=affine)

        max_projection_idx = int(torch.argmax(projections).item())
        min_projection_idx = int(torch.argmin(projections).item())
        if superior[max_projection_idx] < superior[min_projection_idx]:
            axis = -axis
            projections = -projections

        bins = torch.round(projections).to(torch.int64)
        sorted_bins, sorted_areas = self._sorted_slice_areas(bins)
        roundness_ratios = self._slice_roundness_ratios(
            coords=coords,
            bins=bins,
            sorted_bins=sorted_bins,
            axis=axis,
        )
        cutoff = self._detect_cutoff_projection(
            sorted_bins=sorted_bins,
            sorted_areas=sorted_areas,
            roundness_ratios=roundness_ratios,
        )
        keep = projections >= cutoff
        kept_coords = coords[keep].to(torch.long)
        if kept_coords.numel() == 0:
            return output

        if output.ndim == LABEL_EXPECTED_DIMS:
            output[kept_coords[:, 0], kept_coords[:, 1], kept_coords[:, 2]] = 1
        else:
            output[0, kept_coords[:, 0], kept_coords[:, 1], kept_coords[:, 2]] = 1
        return output

    def _extract_mask(self, femur_tensor: Tensor) -> tuple[Tensor, Tensor]:
        if femur_tensor.ndim == LABEL_EXPECTED_DIMS:
            mask = femur_tensor > 0
            output = torch.zeros_like(femur_tensor)
            return mask, output
        if femur_tensor.ndim == LABEL_EXPECTED_DIMS + 1 and femur_tensor.shape[0] == LABEL_CHANNELS:
            mask = femur_tensor[0] > 0
            output = torch.zeros_like(femur_tensor)
            return mask, output
        msg = "Femur mask must have shape (D, H, W) or (1, D, H, W)."
        raise ValueError(msg)

    def _principal_axis(self, coords: Tensor) -> Tensor:
        centered = coords - coords.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        return axis / torch.linalg.norm(axis)

    def _superior_coordinates(self, coords: Tensor, *, affine: np.ndarray) -> Tensor:
        affine_tensor = torch.as_tensor(affine, dtype=torch.float64, device=coords.device)
        homogeneous = torch.cat(
            [coords, torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)],
            dim=1,
        )
        world = homogeneous @ affine_tensor.T
        return world[:, WORLD_SUPERIOR_AXIS]

    def _sorted_slice_areas(self, bins: Tensor) -> tuple[Tensor, Tensor]:
        unique_bins, counts = torch.unique(bins, return_counts=True)
        order = torch.argsort(unique_bins, descending=True)
        sorted_bins = unique_bins[order]
        sorted_areas = counts[order].to(torch.float64)
        return sorted_bins, sorted_areas

    def _slice_roundness_ratios(
        self,
        *,
        coords: Tensor,
        bins: Tensor,
        sorted_bins: Tensor,
        axis: Tensor,
    ) -> Tensor:
        basis_u, basis_v = self._orthogonal_plane_basis(axis)
        ratios = torch.ones(sorted_bins.shape[0], dtype=torch.float64, device=coords.device)
        for idx, bin_value in enumerate(sorted_bins):
            slice_coords = coords[bins == bin_value]
            ratios[idx] = self._ellipse_axis_ratio(
                slice_coords=slice_coords,
                basis_u=basis_u,
                basis_v=basis_v,
            )
        return ratios

    def _orthogonal_plane_basis(self, axis: Tensor) -> tuple[Tensor, Tensor]:
        axis_unit = axis / torch.linalg.norm(axis)
        reference = torch.tensor([1.0, 0.0, 0.0], dtype=axis.dtype, device=axis.device)
        if torch.abs(torch.dot(axis_unit, reference)) > PLANE_REFERENCE_PARALLEL_THRESHOLD:
            reference = torch.tensor([0.0, 1.0, 0.0], dtype=axis.dtype, device=axis.device)

        basis_u = reference - torch.dot(reference, axis_unit) * axis_unit
        basis_u_norm = torch.linalg.norm(basis_u)
        if float(basis_u_norm.item()) <= ROUNDNESS_EPSILON:
            reference = torch.tensor([0.0, 0.0, 1.0], dtype=axis.dtype, device=axis.device)
            basis_u = reference - torch.dot(reference, axis_unit) * axis_unit
            basis_u_norm = torch.linalg.norm(basis_u)
        basis_u = basis_u / basis_u_norm
        basis_v = torch.linalg.cross(axis_unit, basis_u)
        basis_v = basis_v / torch.linalg.norm(basis_v)
        return basis_u, basis_v

    def _ellipse_axis_ratio(
        self,
        *,
        slice_coords: Tensor,
        basis_u: Tensor,
        basis_v: Tensor,
    ) -> float:
        if slice_coords.shape[0] < MIN_ROUNDNESS_SLICE_POINTS:
            return 1.0

        centered = slice_coords - slice_coords.mean(dim=0, keepdim=True)
        in_plane = torch.stack([centered @ basis_u, centered @ basis_v], dim=1)
        _, singular_values, _ = torch.linalg.svd(in_plane, full_matrices=False)
        if singular_values.numel() < ELLIPSE_AXES_COUNT:
            return float("inf")
        major_axis = float(singular_values[0].item())
        minor_axis = float(singular_values[1].item())
        if minor_axis <= ROUNDNESS_EPSILON:
            return float("inf")
        return major_axis / minor_axis

    def _detect_cutoff_projection(
        self,
        *,
        sorted_bins: Tensor,
        sorted_areas: Tensor,
        roundness_ratios: Tensor,
    ) -> float:
        if sorted_bins.numel() == 0:
            return 0.0

        area_idx = self._detect_area_cutoff_index(sorted_areas)
        roundness_idx = self._detect_roundness_cutoff_index(roundness_ratios)
        candidate_indices = [idx for idx in (area_idx, roundness_idx) if idx is not None]
        if not candidate_indices:
            return float(sorted_bins[-1].item())
        cutoff_idx = min(candidate_indices)
        return float(sorted_bins[cutoff_idx].item())

    def _detect_area_cutoff_index(self, sorted_areas: Tensor) -> int | None:
        if sorted_areas.numel() <= self.min_slices_from_top:
            return None

        running_max = 0.0
        consecutive = 0
        first_bad_idx: int | None = None
        for idx, area_tensor in enumerate(sorted_areas):
            area = float(area_tensor.item())
            running_max = max(running_max, area)
            if idx < self.min_slices_from_top:
                consecutive = 0
                first_bad_idx = None
                continue
            if area <= self.area_ratio_threshold * running_max:
                if first_bad_idx is None:
                    first_bad_idx = idx
                consecutive += 1
                if consecutive >= self.confirm_slices and first_bad_idx is not None:
                    return max(first_bad_idx - 1, 0)
            else:
                consecutive = 0
                first_bad_idx = None
        return None

    def _detect_roundness_cutoff_index(self, roundness_ratios: Tensor) -> int | None:
        if roundness_ratios.numel() <= self.min_slices_from_top:
            return None

        consecutive = 0
        first_bad_idx: int | None = None
        for idx, ratio_tensor in enumerate(roundness_ratios):
            ratio = float(ratio_tensor.item())
            if idx < self.min_slices_from_top:
                consecutive = 0
                first_bad_idx = None
                continue
            if ratio > self.roundness_ratio_threshold:
                if first_bad_idx is None:
                    first_bad_idx = idx
                consecutive += 1
                if consecutive >= self.roundness_confirm_slices and first_bad_idx is not None:
                    return max(first_bad_idx - self.roundness_backstep_slices, 0)
            else:
                consecutive = 0
                first_bad_idx = None
        return None

    def _largest_connected_component(self, mask: Tensor) -> Tensor:
        mask_np = mask.detach().cpu().numpy().astype(bool, copy=False)
        if not mask_np.any():
            return mask

        largest_coords = self._largest_component_coords(mask_np)

        output_np = np.zeros_like(mask_np, dtype=np.bool_)
        if largest_coords:
            largest_array = np.asarray(largest_coords, dtype=np.int64)
            output_np[largest_array[:, 0], largest_array[:, 1], largest_array[:, 2]] = True
        output = torch.from_numpy(output_np).to(device=mask.device)
        return output

    def _largest_component_coords(self, mask_np: np.ndarray) -> list[tuple[int, int, int]]:
        visited = np.zeros_like(mask_np, dtype=bool)
        largest_coords: list[tuple[int, int, int]] = []
        for seed_z, seed_y, seed_x in np.argwhere(mask_np):
            z_idx, y_idx, x_idx = int(seed_z), int(seed_y), int(seed_x)
            if visited[z_idx, y_idx, x_idx]:
                continue
            component_coords = self._flood_component(
                (z_idx, y_idx, x_idx),
                mask_np=mask_np,
                visited=visited,
            )
            if len(component_coords) > len(largest_coords):
                largest_coords = component_coords
        return largest_coords

    def _flood_component(
        self,
        seed: tuple[int, int, int],
        *,
        mask_np: np.ndarray,
        visited: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        queue: deque[tuple[int, int, int]] = deque([seed])
        visited[seed] = True
        component_coords: list[tuple[int, int, int]] = []
        while queue:
            z_idx, y_idx, x_idx = queue.popleft()
            component_coords.append((z_idx, y_idx, x_idx))
            for nz, ny, nx in self._iter_neighbor_coords(
                z_idx,
                y_idx,
                x_idx,
                shape=mask_np.shape,
            ):
                if visited[nz, ny, nx] or not mask_np[nz, ny, nx]:
                    continue
                visited[nz, ny, nx] = True
                queue.append((nz, ny, nx))
        return component_coords

    def _iter_neighbor_coords(
        self,
        z_idx: int,
        y_idx: int,
        x_idx: int,
        *,
        shape: tuple[int, int, int],
    ) -> list[tuple[int, int, int]]:
        z_dim, y_dim, x_dim = shape
        neighbors: list[tuple[int, int, int]] = []
        for dz, dy, dx in NEIGHBOR_OFFSETS_26:
            nz = z_idx + dz
            ny = y_idx + dy
            nx = x_idx + dx
            if nz < 0 or ny < 0 or nx < 0:
                continue
            if nz >= z_dim or ny >= y_dim or nx >= x_dim:
                continue
            neighbors.append((nz, ny, nx))
        return neighbors

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
