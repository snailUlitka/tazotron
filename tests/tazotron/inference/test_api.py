from __future__ import annotations

from base64 import b64encode
from io import BytesIO

import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from tazotron.inference.api import app
from tazotron.inference.model import BinaryImageClassifier
from tazotron.inference.service import (
    InferenceFacade,
    RegisteredModel,
    RegisteredModelMetadata,
    get_inference_facade,
)
from tazotron.inference.settings import InferenceSettings


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: tuple[float, float]) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor([logits], dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits.repeat(inputs.shape[0], 1)


def _make_image_base64() -> str:
    image = Image.new("L", (4, 4))
    for x in range(4):
        for y in range(4):
            image.putpixel((x, y), x * 32 + y * 16)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    settings = InferenceSettings(
        model_external_id="model-v1",
        model_display_name="Test Model",
        model_version="test-version",
        model_name="resnet18",
        image_size=8,
    )
    facade = InferenceFacade(settings)
    classifier = BinaryImageClassifier(
        model=FixedLogitModel((0.2, 1.4)),
        settings=settings,
    )
    registered_model = RegisteredModel(
        external_id="model-v1",
        name="Test Model",
        version="test-version",
        accuracy=0.85,
        loss=0.34,
        classifier=classifier,
    )
    metadata = RegisteredModelMetadata(
        external_id="model-v1",
        name="Test Model",
        version="test-version",
        accuracy=0.85,
        loss=0.34,
        checkpoint={"model_state_dict": {}},
        resolved_model_name="resnet18",
    )
    monkeypatch.setattr(facade, "_load_model_metadata", lambda: metadata)
    monkeypatch.setattr(facade, "_build_registered_model", lambda _: registered_model)
    app.dependency_overrides[get_inference_facade] = lambda: facade
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestInferenceApi:
    @pytest.mark.fast
    def test_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.fast
    def test_models_endpoint(self, client: TestClient) -> None:
        response = client.get("/models")

        assert response.status_code == 200
        assert response.json() == [
            {
                "externalId": "model-v1",
                "name": "Test Model",
                "version": "test-version",
            }
        ]

    @pytest.mark.fast
    def test_model_metrics_endpoint(self, client: TestClient) -> None:
        response = client.get("/models/model-v1/metrics")

        assert response.status_code == 200
        assert response.json() == {
            "externalId": "model-v1",
            "accuracy": 0.85,
            "loss": 0.34,
        }

    @pytest.mark.fast
    def test_classify_endpoint(self, client: TestClient) -> None:
        response = client.post(
            "/classify",
            json={"modelExternalId": "model-v1", "imageBase64": _make_image_base64()},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "success"
        assert payload["diagnosis"] == "positive"
        assert 0.5 < payload["probability"] < 1.0
        assert payload["errorMessage"] is None

    @pytest.mark.fast
    def test_classify_unknown_model_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/classify",
            json={"modelExternalId": "missing-model", "imageBase64": _make_image_base64()},
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Model 'missing-model' was not found"

    @pytest.mark.fast
    def test_classify_rejects_invalid_base64(self, client: TestClient) -> None:
        response = client.post(
            "/classify",
            json={"modelExternalId": "model-v1", "imageBase64": "!!!"},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "imageBase64 must contain a valid base64 payload"

    @pytest.mark.fast
    def test_classify_rejects_invalid_image_payload(self, client: TestClient) -> None:
        response = client.post(
            "/classify",
            json={
                "modelExternalId": "model-v1",
                "imageBase64": b64encode(b"not-an-image").decode("ascii"),
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "imageBase64 must decode to a valid image"

    @pytest.mark.fast
    def test_models_returns_503_when_model_loading_fails(self) -> None:
        settings = InferenceSettings(model_external_id="model-v1")
        facade = InferenceFacade(settings)

        def _boom() -> RegisteredModelMetadata:
            raise HTTPException(status_code=503, detail="checkpoint is unavailable")

        app.dependency_overrides[get_inference_facade] = lambda: facade
        try:
            facade._load_model_metadata = _boom  # type: ignore[method-assign]
            with TestClient(app) as test_client:
                models_response = test_client.get("/models")
                response = test_client.get("/models/model-v1/metrics")
        finally:
            app.dependency_overrides.clear()

        assert models_response.status_code == 200
        assert models_response.json() == [
            {
                "externalId": "model-v1",
                "name": settings.model_display_name,
                "version": settings.clearml_task_name,
            }
        ]
        assert response.status_code == 503
        assert response.json()["detail"] == "checkpoint is unavailable"
