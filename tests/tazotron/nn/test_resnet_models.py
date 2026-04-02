"""Tests for custom ResNet-style classifiers."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from tazotron.nn import TrainingConfig
from tazotron.nn._resnet import ResNetClassifierBase
from tazotron.nn.resnet18 import ResNet18
from tazotron.nn.resnet50 import ResNet50


class DictDataset(Dataset[dict[str, torch.Tensor]]):
    """Small synthetic dataset for smoke tests."""

    def __init__(self, size: int = 8) -> None:
        self.samples = [
            {
                "drr": torch.rand(1, 16, 16),
                "label_combined_femoral_head": torch.tensor(index % 2, dtype=torch.long),
            }
            for index in range(size)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


@pytest.mark.fast
@pytest.mark.parametrize("model_cls", [ResNet18, ResNet50])
@pytest.mark.parametrize("channels", [1, 3])
def test_model_forward_supports_grayscale_and_rgb(
    model_cls: type[ResNetClassifierBase],
    channels: int,
) -> None:
    model = model_cls(num_classes=2)
    outputs = model(torch.rand(2, channels, 32, 32))
    assert outputs.shape == (2, 2)


@pytest.mark.fast
def test_prepare_inputs_squeezes_and_expands_grayscale() -> None:
    inputs = torch.rand(2, 1, 1, 16, 20)
    prepared = ResNetClassifierBase._prepare_inputs(inputs, image_size=24)
    assert prepared.shape == (2, 3, 24, 24)


@pytest.mark.fast
def test_prepare_inputs_rejects_invalid_channel_count() -> None:
    with pytest.raises(ValueError, match="Expected 1 or 3 channels"):
        ResNetClassifierBase._prepare_inputs(torch.rand(2, 2, 16, 16), image_size=16)


@pytest.mark.fast
@pytest.mark.parametrize("model_cls", [ResNet18, ResNet50])
def test_ensure_model_compatible_checks_output_size(model_cls: type[ResNetClassifierBase]) -> None:
    model = model_cls(num_classes=2)
    with pytest.raises(ValueError, match="does not match model output features"):
        ResNetClassifierBase._ensure_model_compatible(model, TrainingConfig(num_classes=3))


@pytest.mark.fast
@pytest.mark.parametrize("model_cls", [ResNet18, ResNet50])
def test_train_and_test_smoke(model_cls: type[ResNetClassifierBase]) -> None:
    model = model_cls(num_classes=2)
    loader = DataLoader(DictDataset(), batch_size=2, shuffle=False)
    config = TrainingConfig(
        num_classes=2,
        folds=2,
        num_epochs=1,
        image_size=16,
        device="cpu",
        num_workers=0,
    )

    train_result = model_cls.train_(model, loader, config)
    test_result = model_cls.test_(model, loader, config)

    assert train_result["fold_count"] == 2
    assert len(train_result["folds"]) == 2
    assert "mean_best_val_accuracy" in train_result
    assert set(test_result) == {"loss", "accuracy", "samples"}
