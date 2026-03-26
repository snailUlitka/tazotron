from __future__ import annotations

from io import BytesIO

import pytest
import torch
from PIL import Image

from tazotron.inference.model import (
    BinaryImageClassifier,
    BinaryImagePreprocessor,
    extract_metrics,
    minmax_normalize,
    resolve_model_name,
)
from tazotron.inference.settings import InferenceSettings


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: tuple[float, float]) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor([logits], dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits.repeat(inputs.shape[0], 1)


def _make_png_bytes() -> bytes:
    image = Image.new("L", (4, 4))
    for x in range(4):
        for y in range(4):
            image.putpixel((x, y), x * 32 + y * 16)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestInferenceModel:
    @pytest.mark.fast
    def test_minmax_normalize_returns_zeros_for_constant_tensor(self) -> None:
        tensor = torch.ones((1, 2, 2), dtype=torch.float32)

        normalized = minmax_normalize(tensor)

        assert torch.count_nonzero(normalized) == 0

    @pytest.mark.fast
    def test_preprocessor_builds_three_channel_batched_tensor(self) -> None:
        settings = InferenceSettings(model_name="resnet18", image_size=8)
        preprocessor = BinaryImagePreprocessor(settings)

        prepared = preprocessor.prepare(_make_png_bytes())

        assert prepared.shape == (1, 3, 8, 8)
        assert prepared.dtype == torch.float32

    @pytest.mark.fast
    def test_classifier_predicts_positive_probability(self) -> None:
        settings = InferenceSettings(model_name="resnet18", image_size=8, classification_threshold=0.5)
        classifier = BinaryImageClassifier(
            model=FixedLogitModel((0.1, 1.1)),
            settings=settings,
        )

        diagnosis, probability = classifier.predict(_make_png_bytes())

        assert diagnosis == "positive"
        assert 0.5 < probability < 1.0

    @pytest.mark.fast
    def test_resolve_model_name_prefers_checkpoint_config(self) -> None:
        settings = InferenceSettings(model_name="resnet50")
        checkpoint = {"config": {"model_name": "resnet18"}}

        assert resolve_model_name(settings, checkpoint) == "resnet18"

    @pytest.mark.fast
    def test_extract_metrics_reads_accuracy_and_loss(self) -> None:
        checkpoint = {"val_metrics": {"accuracy": 0.85, "loss": 0.34}}

        accuracy, loss = extract_metrics(checkpoint)

        assert accuracy == pytest.approx(0.85)
        assert loss == pytest.approx(0.34)
