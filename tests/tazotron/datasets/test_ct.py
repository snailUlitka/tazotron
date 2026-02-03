import tempfile
from pathlib import Path

import pytest
import torch
import torchio as tio

from tazotron.datasets.ct import CTDataset


def _write_ct(path: Path) -> None:
    image = tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2), affine=torch.eye(4))
    image.save(path)


def _write_label(path: Path, tensor: torch.Tensor | None = None) -> None:
    if tensor is None:
        tensor = torch.zeros(1, 2, 2, 2, dtype=torch.int16)
    label = tio.LabelMap(tensor=tensor, affine=torch.eye(4))
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
    def test_allows_missing_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "case_001"
            case_dir.mkdir(parents=True)
            _write_ct(case_dir / "ct.nii.gz")
            left_tensor = torch.zeros(1, 2, 2, 2, dtype=torch.int16)
            left_tensor[0, 0, 0, 0] = 1
            _write_label(case_dir / "label_femoral_head_left.nii.gz", left_tensor)

            dataset = CTDataset(temp_dir)

            subject = dataset[0]

            assert subject["label_femoral_head_right"] is None
            assert torch.any(subject["label_combined_femoral_head"].data == 1)
            assert not torch.any(subject["label_combined_femoral_head"].data == 2)

    @pytest.mark.fast
    def test_allows_missing_default_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "case_001"
            case_dir.mkdir(parents=True)
            _write_ct(case_dir / "ct.nii.gz")
            _write_label(case_dir / "left_label.nii.gz")
            _write_label(case_dir / "right_label.nii.gz")

            dataset = CTDataset(temp_dir)

            subject = dataset[0]

            assert subject["label_femoral_head_left"] is None
            assert subject["label_femoral_head_right"] is None
            assert not torch.any(subject["label_combined_femoral_head"].data > 0)

    @pytest.mark.fast
    def test_loads_masks_from_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir) / "case_001"
            case_dir.mkdir(parents=True)
            _write_ct(case_dir / "ct.nii.gz")

            left_tensor = torch.zeros(1, 2, 2, 2, dtype=torch.int16)
            right_tensor = torch.zeros(1, 2, 2, 2, dtype=torch.int16)
            left_tensor[0, 0, 0, 0] = 1
            right_tensor[0, 1, 1, 1] = 1
            extra_tensor = torch.zeros(1, 2, 2, 2, dtype=torch.int16)
            left_name = "left_custom.nii.gz"
            right_name = "right_custom.nii.gz"
            extra_name = "extra_mask.nii.gz"
            _write_label(case_dir / left_name, left_tensor)
            _write_label(case_dir / right_name, right_tensor)
            _write_label(case_dir / extra_name, extra_tensor)
            _write_label(case_dir / "hip_left.nii.gz.seg.nrrd")
            _write_label(case_dir / "hip_right.nii.gz.seg.nrrd")

            dataset = CTDataset(
                temp_dir,
                mask_paths={
                    "label_femoral_head_left": left_name,
                    "label_femoral_head_right": right_name,
                    "extra_mask": extra_name,
                },
            )

            subject = dataset[0]

            assert "label_femoral_head_left" in subject
            assert "label_femoral_head_right" in subject
            assert "extra_mask" in subject
            assert torch.equal(subject["label_femoral_head_left"].data, left_tensor)
            assert torch.equal(subject["label_femoral_head_right"].data, right_tensor)
            assert torch.any(subject["label_combined_femoral_head"].data == 1)
            assert torch.any(subject["label_combined_femoral_head"].data == 2)
