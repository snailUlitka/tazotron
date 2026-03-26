"""Model loading and image preprocessing for runtime inference."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import timm
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms as T

from tazotron.inference.settings import InferenceSettings

EPS = 1e-6


def minmax_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to the [0, 1] range."""
    tensor = tensor.float()
    min_value = tensor.min()
    max_value = tensor.max()
    if float(max_value - min_value) < EPS:
        return torch.zeros_like(tensor, dtype=torch.float32)
    return (tensor - min_value) / (max_value - min_value)


def ensure_three_channels(image: torch.Tensor) -> torch.Tensor:
    """Expand a grayscale image tensor to 3 channels."""
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim != 3:
        msg = f"Expected CHW tensor, got shape {tuple(image.shape)}"
        raise ValueError(msg)
    if image.shape[0] == 1:
        return image.repeat(3, 1, 1)
    if image.shape[0] == 3:
        return image
    msg = f"Expected 1 or 3 channels, got {image.shape[0]}"
    raise ValueError(msg)


class BinaryImagePreprocessor:
    """Apply the evaluation pipeline from the training notebook."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.transform = T.Compose(
            [
                T.Resize((settings.image_size, settings.image_size), antialias=True),
                T.Lambda(ensure_three_channels),
                T.Normalize(mean=settings.normalize_mean, std=settings.normalize_std),
            ]
        )

    def prepare(self, image_bytes: bytes) -> torch.Tensor:
        """Decode image bytes and convert them into a model-ready tensor."""
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                tensor = T.ToTensor()(image.convert("L"))
        except (UnidentifiedImageError, OSError) as error:
            msg = "imageBase64 must decode to a valid image"
            raise ValueError(msg) from error

        normalized = minmax_normalize(tensor)
        return self.transform(normalized).unsqueeze(0)


def build_binary_model(model_name: str, num_classes: int) -> nn.Module:
    """Create a timm model with a binary classifier head."""
    model = timm.create_model(model_name, pretrained=False)
    if not hasattr(model, "reset_classifier"):
        msg = f"{model_name} does not expose reset_classifier()."
        raise ValueError(msg)
    model.reset_classifier(num_classes=num_classes)
    return model


def resolve_model_name(settings: InferenceSettings, checkpoint: object) -> str:
    """Pick the runtime model name from checkpoint config or settings."""
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            checkpoint_model_name = config.get("model_name")
            if isinstance(checkpoint_model_name, str) and checkpoint_model_name:
                return checkpoint_model_name
    return settings.model_name


def extract_model_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract the model state dict from a checkpoint payload."""
    if isinstance(checkpoint, dict):
        nested = checkpoint.get("model_state_dict")
        if isinstance(nested, dict):
            return nested
        if checkpoint and all(isinstance(key, str) for key in checkpoint) and all(
            isinstance(value, torch.Tensor) for value in checkpoint.values()
        ):
            return checkpoint
    msg = "Checkpoint must contain a model_state_dict dictionary"
    raise ValueError(msg)


def extract_metrics(checkpoint: object) -> tuple[float, float]:
    """Extract accuracy and loss from checkpoint validation metrics."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")
    val_metrics = checkpoint.get("val_metrics")
    if not isinstance(val_metrics, dict):
        raise ValueError("Checkpoint is missing val_metrics")
    accuracy = val_metrics.get("accuracy")
    loss = val_metrics.get("loss")
    if not isinstance(accuracy, (int, float)) or not isinstance(loss, (int, float)):
        raise ValueError("Checkpoint val_metrics must contain numeric accuracy and loss")
    return float(accuracy), float(loss)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load a checkpoint from disk."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        msg = f"Expected checkpoint dictionary, got {type(checkpoint)}"
        raise ValueError(msg)
    return checkpoint


class BinaryImageClassifier:
    """Thin wrapper around a binary timm model."""

    def __init__(
        self,
        *,
        model: nn.Module,
        settings: InferenceSettings,
    ) -> None:
        self.device = torch.device(settings.model_device)
        self.model = model.to(self.device)
        self.model.eval()
        self.threshold = settings.classification_threshold
        self.preprocessor = BinaryImagePreprocessor(settings)

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        """Predict the positive class probability and binary diagnosis."""
        inputs = self.preprocessor.prepare(image_bytes).to(self.device)
        with torch.inference_mode():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            msg = f"Expected logits for 2 classes, got shape {tuple(probabilities.shape)}"
            raise ValueError(msg)
        positive_probability = float(probabilities[0, 1].item())
        diagnosis = "positive" if positive_probability >= self.threshold else "negative"
        return diagnosis, positive_probability
