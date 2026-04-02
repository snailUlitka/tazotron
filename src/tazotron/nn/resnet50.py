"""Larger ResNet-style classifier for XR inputs."""

from __future__ import annotations

from tazotron.nn._resnet import ResNetClassifierBase


class ResNet50(ResNetClassifierBase):
    """Larger ResNet-style CNN (no residual blocks) for grayscale XR inputs."""

    stage_channels = (
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    )
    pool_after_stages = (0, 1, 2)
    classifier_hidden_features = 256


__all__ = ["ResNet50"]
