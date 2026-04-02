"""Compact ResNet-style classifier for XR inputs."""

from __future__ import annotations

from tazotron.nn._resnet import ResNetClassifierBase, TrainingConfig


class ResNet18(ResNetClassifierBase):
    """Compact ResNet-style CNN (no residual blocks) tailored for grayscale XR inputs."""

    stage_channels = ((32, 32), (64, 64), (128,))
    pool_after_stages = (0, 1)
    classifier_hidden_features = 64


__all__ = ["ResNet18", "TrainingConfig"]
