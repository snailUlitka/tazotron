import tempfile
from pathlib import Path

import pytest
import torch
import torchio as tio

from tazotron.datasets.ct import CTDataset


def _write_ct(path: Path) -> None:
    image = tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2), affine=torch.eye(4))
    image.save(path)


def _write_label(path: Path) -> None:
    label = tio.LabelMap(tensor=torch.zeros(1, 2, 2, 2, dtype=torch.int16), affine=torch.eye(4))
    label.save(path)


class TestCTDataset:
    @pytest.mark.fast
    def test_raises_for_non_directory_data_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "not_a_dir.txt"
            data_path.write_text("x", encoding="utf-8")

            with pytest.raises(ValueError, match="not a directory"):
                CTDataset(data_path)

    @pytest.mark.fast
    def test_raises_when_ct_path_is_not_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "case_001"
            case_dir.mkdir(parents=True)
            (case_dir / "ct.nii.gz").mkdir()

            with pytest.raises(ValueError, match="No CT files found"):
                CTDataset(temp_dir)

    @pytest.mark.fast
    def test_raises_when_no_ct_files_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "case_001"
            case_dir.mkdir(parents=True)

            with pytest.raises(ValueError, match="No CT files found"):
                CTDataset(temp_dir)

    @pytest.mark.fast
    def test_raises_when_only_one_label_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "case_001"
            case_dir.mkdir(parents=True)
            _write_ct(case_dir / "ct.nii.gz")
            _write_label(case_dir / "label_femoral_head_left.nii.gz")

            dataset = CTDataset(temp_dir)

            with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
                _ = dataset[0]

    @pytest.mark.fast
    def test_raises_when_label_names_do_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "case_001"
            case_dir.mkdir(parents=True)
            _write_ct(case_dir / "ct.nii.gz")
            _write_label(case_dir / "left_label.nii.gz")
            _write_label(case_dir / "right_label.nii.gz")

            dataset = CTDataset(temp_dir)

            with pytest.raises((FileNotFoundError, OSError, RuntimeError)):
                _ = dataset[0]
