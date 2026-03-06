"""Neural network architectures."""

# Re-export core neural network module components.
from tazotron.nn.resnet18 import ResNet18, TrainingConfig

# Public module API.
__all__ = ["ResNet18", "TrainingConfig"]
