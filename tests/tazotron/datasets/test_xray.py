import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tazotron.datasets.xray import XrayDataset


def _write_tiff(path: Path, array: np.ndarray) -> None:
    image = Image.fromarray(array)
    image.save(path)


class TestXrayDataset:
    @pytest.mark.fast
    def test_raises_when_no_tiff_files_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            (data_dir / "note.txt").write_text("x", encoding="utf-8")

            with pytest.raises(ValueError, match="No TIFF files found"):
                XrayDataset(data_dir)

    @pytest.mark.fast
    def test_loads_tiff_as_grayscale_tensor_without_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            array = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16)
            _write_tiff(data_dir / "sample.tiff", array)

            dataset = XrayDataset(data_dir)
            tensor = dataset[0]

            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (1, 2, 2)
            assert tensor.dtype == torch.float32
            assert torch.isclose(tensor[0, 0, 0], torch.tensor(0.0))
            assert torch.isclose(tensor[0, 0, 1], torch.tensor(1.0))

    @pytest.mark.fast
    def test_applies_transform_to_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            array = np.full((2, 2), 1000, dtype=np.uint16)
            _write_tiff(data_dir / "sample.tiff", array)

            dataset = XrayDataset(data_dir, transform=lambda x: x * 0)
            tensor = dataset[0]

            assert torch.count_nonzero(tensor) == 0

