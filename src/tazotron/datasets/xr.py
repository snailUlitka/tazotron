"""Datasets for DRR/X-ray tensors stored on disk."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class XrayTensorDataset(Dataset[dict[str, torch.Tensor]]):
    """Loads DRR images stored as TIFF (uint16, lossless) or PyTorch tensors (.pt)."""

    def __init__(  # noqa: PLR0913
        self,
        xr_dir: str | Path,
        *,
        key: str = "drr",
        backend: str = "tiff",  # "tiff" | "pt"
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
        extensions: Iterable[str] = (".tiff", ".tif", ".pt"),
        map_location: str | torch.device | None = None,
    ) -> None:
        self.xr_dir = Path(xr_dir)
        self.key = key
        self.backend = backend
        self.transform = transform
        self.extensions = tuple(extensions)
        self.map_location = map_location
        self.paths = self._collect_paths()
        if not self.paths:
            msg = f"No XR files found in {self.xr_dir}"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one tensor sample."""
        path = self.paths[index]
        if self.backend == "pt" or path.suffix == ".pt":
            tensor = torch.load(path, map_location=self.map_location)
            sample = {self.key: tensor, "path": str(path)}
        else:
            tensor = self._load_tiff(path)
            sample = {self.key: tensor, "path": str(path)}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _collect_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.xr_dir.iterdir()
            if path.is_file() and any(path.suffix == ext for ext in self.extensions)
        )

    def _load_tiff(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            image = img
            if image.mode not in {"I;16", "I"}:
                warnings.warn(
                    f"Converting image {path} to 16-bit; original mode={image.mode}",
                    stacklevel=2,
                )
                image = image.convert("I;16")
            array = np.array(image, dtype=np.uint16, copy=False)
            tensor = torch.from_numpy(array).to(torch.float32) / 65535.0
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            return tensor
