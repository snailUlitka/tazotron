"""Datasets for CT volumes stored on disk."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchio as tio
from diffdrr.data import read
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

FEMORAL_HEAD_LEFT = "label_femoral_head_left"
FEMORAL_HEAD_RIGHT = "label_femoral_head_right"
COMBINED_FEMORAL_HEAD = "label_combined_femoral_head"
FEMUR_LEFT = "label_femur_left"
FEMUR_RIGHT = "label_femur_right"


class CTDataset(Dataset[tio.Subject]):
    """Loads CT volumes (and labelmaps) from a directory of .nii.gz files.

    Returns torchio.Subject objects compatible with downstream transforms such as necrosis injection or DRR rendering.
    Additional label maps can be provided via ``mask_paths`` where keys become subject entries.
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        ct_name: str | Path = "ct.nii.gz",
        mask_paths: Mapping[str, str | Path] | None = None,
        transform: Callable[[tio.Subject], tio.Subject] | None = None,
    ) -> None:
        self.data_path = Path(data_path)

        if not self.data_path.is_dir():
            msg = f"Data path {self.data_path} is not a directory"
            raise ValueError(msg)

        self.ct_name = Path(ct_name)
        self.mask_paths = self._normalize_mask_paths(mask_paths)

        self.transform = transform

        self.paths = self._collect_paths()

        if not self.paths:
            msg = f"No CT files found in {self.data_path}"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.paths)

    @staticmethod
    def save_femoral_head_masks(subject: tio.Subject, output_dir: str | Path) -> None:
        """Save femoral head masks from a subject into the output directory."""
        if not isinstance(subject, tio.Subject):
            msg = "save_femoral_head_masks expects a torchio.Subject."
            raise TypeError(msg)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        left_mask = subject.get(FEMORAL_HEAD_LEFT)
        right_mask = subject.get(FEMORAL_HEAD_RIGHT)

        if left_mask is None or right_mask is None:
            msg = "Both femoral head masks must be present to save."
            raise ValueError(msg)
        if not isinstance(left_mask, tio.LabelMap) or not isinstance(right_mask, tio.LabelMap):
            msg = "Femoral head masks must be torchio.LabelMap instances."
            raise TypeError(msg)

        left_path = output_path / f"{FEMORAL_HEAD_LEFT}.nii.gz"
        right_path = output_path / f"{FEMORAL_HEAD_RIGHT}.nii.gz"
        left_mask.save(left_path)
        right_mask.save(right_path)

    def __getitem__(self, index: int) -> tio.Subject:
        """Load one sample as torchio.Subject."""
        ct_path = self.paths[index]

        label_left_relative = self.mask_paths[FEMORAL_HEAD_LEFT]
        label_right_relative = self.mask_paths[FEMORAL_HEAD_RIGHT]

        label_left_path = ct_path.parent / label_left_relative
        label_right_path = ct_path.parent / label_right_relative

        ct = tio.ScalarImage(ct_path)

        label_left = self._load_labelmap(label_left_path)
        label_right = self._load_labelmap(label_right_path)

        if label_left is not None and label_right is not None and label_left.data.shape != label_right.data.shape:
            msg = "Labelmap shape mismatch for femoral head labels."
            raise ValueError(msg)

        combined = self._build_combined_label(label_left, label_right, ct)
        combined_affine = self._select_affine(label_left, label_right, ct)

        label = tio.LabelMap(tensor=combined, affine=combined_affine)
        subject = read(volume=ct, labelmap=label, label=label)
        if "label" in subject:
            subject[COMBINED_FEMORAL_HEAD] = subject["label"]
            del subject["label"]
        else:
            subject[COMBINED_FEMORAL_HEAD] = label
        self._append_masks(subject, label_left, label_right, ct_path.parent)

        if self.transform:
            subject = self.transform(subject)

        return subject

    def _collect_paths(self) -> list[Path]:
        ct_paths: list[Path] = []

        for entry in sorted(self.data_path.iterdir()):
            if not entry.is_dir():
                continue

            ct_path = entry / str(self.ct_name)

            if ct_path.is_file():
                ct_paths.append(ct_path)

        return ct_paths

    def _normalize_mask_paths(
        self,
        mask_paths: Mapping[str, str | Path] | None,
    ) -> dict[str, Path]:
        default_masks: dict[str, str | Path] = {
            FEMORAL_HEAD_LEFT: "label_femoral_head_left.nii.gz",
            FEMORAL_HEAD_RIGHT: "label_femoral_head_right.nii.gz",
            FEMUR_LEFT: "femur_left.nii.gz.seg.nrrd",
            FEMUR_RIGHT: "femur_right.nii.gz.seg.nrrd",
        }

        if mask_paths:
            default_masks.update(mask_paths)

        return {key: Path(path) for key, path in default_masks.items()}

    def _append_masks(
        self,
        subject: tio.Subject,
        label_left: tio.LabelMap | None,
        label_right: tio.LabelMap | None,
        base_dir: Path,
    ) -> None:
        for key, rel_path in self.mask_paths.items():
            if key == FEMORAL_HEAD_LEFT:
                subject[key] = label_left
                continue
            if key == FEMORAL_HEAD_RIGHT:
                subject[key] = label_right
                continue
            mask_path = base_dir / rel_path
            subject[key] = self._load_labelmap(mask_path)

    def _load_labelmap(self, path: Path) -> tio.LabelMap | None:
        if not path.is_file():
            return None
        return tio.LabelMap(path)

    def _build_combined_label(
        self,
        label_left: tio.LabelMap | None,
        label_right: tio.LabelMap | None,
        ct: tio.ScalarImage,
    ) -> torch.Tensor:
        if label_left is not None:
            combined = torch.zeros_like(label_left.data)
        elif label_right is not None:
            combined = torch.zeros_like(label_right.data)
        else:
            device = ct.data.device
            combined = torch.zeros(ct.data.shape, dtype=torch.int16, device=device)

        if label_left is not None:
            combined = torch.where(label_left.data > 0, torch.ones_like(combined), combined)
        if label_right is not None:
            combined = torch.where(label_right.data > 0, torch.full_like(combined, 2), combined)

        return combined

    def _select_affine(
        self,
        label_left: tio.LabelMap | None,
        label_right: tio.LabelMap | None,
        ct: tio.ScalarImage,
    ) -> torch.Tensor:
        if label_left is not None:
            return label_left.affine
        if label_right is not None:
            return label_right.affine
        return ct.affine
