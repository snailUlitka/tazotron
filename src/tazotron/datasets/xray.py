"""Minimal TIFF-backed dataset for X-ray images."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class XrayDataset(Dataset[torch.Tensor]):
    """Loads X-ray images from TIFF files and returns grayscale tensors."""

    def __init__(
        self,
        xr_dir: str | Path,
        *,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.xr_dir = Path(xr_dir)
        self.transform = transform
        self.paths = self._collect_paths()

        if not self.paths:
            msg = f"No TIFF files found in {self.xr_dir}"
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load one TIFF sample."""
        path = self.paths[index]
        tensor = self._load_tiff(path)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor

    def _collect_paths(self) -> list[Path]:
        return sorted(path for path in self.xr_dir.iterdir() if path.is_file() and path.suffix == ".tiff")

    def _load_tiff(self, path: Path) -> torch.Tensor:
        """Read a TIFF file and return a normalized grayscale tensor (1, H, W)."""
        with Image.open(path) as img:
            image = img if img.mode in {"I;16", "I"} else img.convert("I;16")
            # Ensure the NumPy buffer is writable before creating a torch view.
            array = np.asarray(image, dtype=np.uint16).copy()

        tensor = torch.from_numpy(array).to(torch.float32) / 65535.0

        return tensor.unsqueeze(0)
