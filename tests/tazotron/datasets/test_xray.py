import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tazotron.datasets.xray import XrayDataset


def _write_pt(path: Path, tensor: torch.Tensor) -> None:
    torch.save(tensor, path)


class TestXrayDataset:
    @pytest.mark.fast
    def test_raises_when_no_pt_files_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "note.txt").write_text("x", encoding="utf-8")

            with pytest.raises(ValueError, match="No .pt files found"):
                XrayDataset(data_dir)

    @pytest.mark.fast
    def test_loads_pt_as_tensor_without_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            tensor = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
            _write_pt(data_dir / "sample.pt", tensor)

            dataset = XrayDataset(data_dir)
            loaded = dataset[0]

            assert isinstance(loaded, torch.Tensor)
            assert loaded.shape == (1, 2, 2)
            assert loaded.dtype == torch.float32
            assert torch.isclose(loaded[0, 0, 0], torch.tensor(0.0))
            assert torch.isclose(loaded[0, 0, 1], torch.tensor(1.0))

    @pytest.mark.fast
    def test_applies_transform_to_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            tensor = torch.full((2, 2), 1000.0, dtype=torch.float32)
            _write_pt(data_dir / "sample.pt", tensor)

            dataset = XrayDataset(data_dir, transform=lambda x: x * 0)
            loaded = dataset[0]

            assert torch.count_nonzero(loaded) == 0

    @pytest.mark.fast
    def test_diff_to_pil_returns_rgb_image(self) -> None:
        xray_a = torch.zeros((1, 2, 2), dtype=torch.float32)
        xray_b = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]], dtype=torch.float32)

        with pytest.deprecated_call():
            image = XrayDataset.diff_to_pil(xray_a, xray_b, quantile=1.0)

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        array = np.array(image)
        assert array.ndim == 3
        assert np.any(array != 255)
