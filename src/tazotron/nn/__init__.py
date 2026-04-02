"""Neural network architectures."""

# Re-export core neural network module components.
from tazotron.nn._resnet import TrainingConfig
from tazotron.nn.resnet18 import ResNet18
from tazotron.nn.resnet50 import ResNet50

# Public module API.
__all__ = ["ResNet18", "ResNet50", "TrainingConfig"]
