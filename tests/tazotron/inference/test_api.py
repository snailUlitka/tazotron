from __future__ import annotations

from base64 import b64encode
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from tazotron.inference.api import app
from tazotron.inference.model import BinaryImageClassifier
from tazotron.inference.schemas import TrainingJobResponse
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


def _make_png_bytes() -> bytes:
    image = Image.new("L", (4, 4), color=128)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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
        checkpoint_path=Path("checkpoint.pt"),
        resolved_model_name="resnet18",
        loader_kind="timm",
    )
    monkeypatch.setattr(facade, "_load_configured_model_metadata", lambda: metadata)
    monkeypatch.setattr(facade, "_build_registered_model", lambda _: registered_model)
    monkeypatch.setattr(facade, "generate_xray_from_ct_request", lambda *_: _make_png_bytes())
    monkeypatch.setattr(
        facade.training_jobs,
        "list_architectures",
        lambda: [
            {
                "key": "resnet18",
                "name": "ResNet18",
                "description": "Compact classifier",
                "requiresManualWeights": False,
            }
        ],
    )
    monkeypatch.setattr(
        facade.training_jobs,
        "start_job",
        lambda *_: TrainingJobResponse(
            jobId="job-001",
            status="queued",
            datasetName="dataset-alpha",
            architectureKey="resnet18",
            learningRate=0.001,
            epochs=5,
            batchSize=4,
            currentEpoch=0,
            totalEpochs=5,
            progressMessage="Queued",
            modelExternalId=None,
            errorMessage=None,
            startedAt=datetime.fromisoformat("2026-04-02T12:00:00+00:00"),
            finishedAt=None,
        ),
    )
    monkeypatch.setattr(
        facade.training_jobs,
        "get_current_job",
        lambda: TrainingJobResponse(
            jobId="job-001",
            status="running",
            datasetName="dataset-alpha",
            architectureKey="resnet18",
            learningRate=0.001,
            epochs=5,
            batchSize=4,
            currentEpoch=2,
            totalEpochs=5,
            progressMessage="Epoch 2/5",
            modelExternalId=None,
            errorMessage=None,
            startedAt=datetime.fromisoformat("2026-04-02T12:00:00+00:00"),
            finishedAt=None,
        ),
    )
    monkeypatch.setattr(
        facade.training_jobs,
        "get_job",
        lambda *_: TrainingJobResponse(
            jobId="job-001",
            status="running",
            datasetName="dataset-alpha",
            architectureKey="resnet18",
            learningRate=0.001,
            epochs=5,
            batchSize=4,
            currentEpoch=2,
            totalEpochs=5,
            progressMessage="Epoch 2/5",
            modelExternalId=None,
            errorMessage=None,
            startedAt=datetime.fromisoformat("2026-04-02T12:00:00+00:00"),
            finishedAt=None,
        ),
    )
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
    def test_ct_to_xray_endpoint(self, client: TestClient) -> None:
        response = client.post(
            "/ct-to-xray",
            json={
                "ctFileBase64": b64encode(b"ct-bytes").decode("ascii"),
                "segmentations": [
                    {"type": "femur_left", "fileBase64": b64encode(b"left").decode("ascii")},
                    {"type": "femur_right", "fileBase64": b64encode(b"right").decode("ascii")},
                ],
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content == _make_png_bytes()

    @pytest.mark.fast
    def test_ct_to_xray_rejects_invalid_base64(self) -> None:
        facade = InferenceFacade(InferenceSettings(model_external_id="model-v1"))
        app.dependency_overrides[get_inference_facade] = lambda: facade
        try:
            with TestClient(app) as test_client:
                response = test_client.post(
                    "/ct-to-xray",
                    json={
                        "ctFileBase64": "!!!",
                        "segmentations": [
                            {"type": "femur_left", "fileBase64": b64encode(b"left").decode("ascii")},
                            {"type": "femur_right", "fileBase64": b64encode(b"right").decode("ascii")},
                        ],
                    },
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 400
        assert response.json()["detail"] == "ctFileBase64 must contain a valid base64 payload"

    @pytest.mark.fast
    def test_training_architectures_endpoint(self, client: TestClient) -> None:
        response = client.get("/training/architectures")

        assert response.status_code == 200
        assert response.json() == [
            {
                "key": "resnet18",
                "name": "ResNet18",
                "description": "Compact classifier",
                "requiresManualWeights": False,
            }
        ]

    @pytest.mark.fast
    def test_start_training_job_endpoint(self, client: TestClient) -> None:
        response = client.post(
            "/training/jobs",
            json={
                "architectureKey": "resnet18",
                "datasetName": "dataset-alpha",
                "learningRate": 0.001,
                "epochs": 5,
                "batchSize": 4,
            },
        )

        assert response.status_code == 200
        assert response.json()["jobId"] == "job-001"

    @pytest.mark.fast
    def test_current_training_job_endpoint(self, client: TestClient) -> None:
        response = client.get("/training/jobs/current")

        assert response.status_code == 200
        assert response.json()["status"] == "running"
        assert response.json()["currentEpoch"] == 2

    @pytest.mark.fast
    def test_models_returns_503_when_model_loading_fails(self) -> None:
        settings = InferenceSettings(model_external_id="model-v1")
        facade = InferenceFacade(settings)

        def _boom() -> RegisteredModelMetadata:
            raise HTTPException(status_code=503, detail="checkpoint is unavailable")

        app.dependency_overrides[get_inference_facade] = lambda: facade
        try:
            facade._load_configured_model_metadata = _boom  # type: ignore[method-assign]
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
