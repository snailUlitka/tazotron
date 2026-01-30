"""Dataset helpers for rendered X-rays."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable

DRR_BATCH_RANK = 4
XRAY_EPS = 1e-6


class XrayDataset(Dataset[Tensor]):
    """Load TIFF X-rays from a directory."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            msg = f"Data path {self.data_path} is not a directory"
            raise ValueError(msg)
        self.transform = transform
        self.paths = self._collect_paths()
        if not self.paths:
            msg = f"No TIFF files found in {self.data_path}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index]
        image = Image.open(path)
        if image.mode != "I;16":
            image = image.convert("I;16")
        array = np.array(image)
        tensor = torch.from_numpy(array).to(torch.float32)
        tensor = self._normalize_tensor(tensor, array.dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor

    @staticmethod
    def to_pil(
        xray: Tensor,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Image.Image:
        """Convert an XR tensor to a Pillow grayscale image."""
        xray = XrayDataset._squeeze_xray(xray)
        if vmin is None:
            vmin = float(xray.min().item())
        if vmax is None:
            vmax = float(xray.max().item())
        denom = max(vmax - vmin, XRAY_EPS)
        scaled = (xray - vmin) / denom
        array = (scaled * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return Image.fromarray(array, mode="L")

    @staticmethod
    def save_pt(xray: Tensor, path: str | Path) -> None:
        """Save an XR tensor as a .pt file without normalization."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        xray = XrayDataset._squeeze_xray(xray).detach().cpu()
        torch.save(xray, output_path)

    @staticmethod
    def diff_to_pil(
        xray_a: Tensor,
        xray_b: Tensor,
        *,
        quantile: float = 0.99,
        vmax: float | None = None,
    ) -> Image.Image:
        """Convert the XR diff (b - a) into a red/white/blue Pillow image."""
        xray_a = XrayDataset._squeeze_xray(xray_a)
        xray_b = XrayDataset._squeeze_xray(xray_b)
        diff = xray_b - xray_a
        abs_diff = diff.abs()
        if vmax is None:
            if abs_diff.numel() == 0:
                vmax = 0.0
            else:
                vmax = float(torch.quantile(abs_diff.flatten(), quantile).item())
        vmax = max(vmax, XRAY_EPS)
        norm = (diff / vmax).clamp(-1.0, 1.0)
        base = 1.0 - norm.abs()
        red = norm.clamp(min=0.0)
        blue = (-norm).clamp(min=0.0)
        rgb = torch.stack([base + red, base, base + blue], dim=-1).clamp(0.0, 1.0)
        array = (rgb * 255).to(torch.uint8).cpu().numpy()
        return Image.fromarray(array, mode="RGB")

    def _collect_paths(self) -> list[Path]:
        return sorted(self.data_path.glob("*.tiff")) + sorted(self.data_path.glob("*.tif"))

    @staticmethod
    def _squeeze_xray(xray: Tensor) -> Tensor:
        if xray.ndim == DRR_BATCH_RANK:
            xray = xray[0, 0]
        elif xray.ndim == 3 and xray.shape[0] == 1:
            xray = xray[0]
        return xray.to(torch.float32)

    @staticmethod
    def _normalize_tensor(tensor: Tensor, dtype: np.dtype) -> Tensor:
        if dtype == np.uint16:
            return tensor / 65535.0
        if dtype == np.uint8:
            return tensor / 255.0
        return tensor
