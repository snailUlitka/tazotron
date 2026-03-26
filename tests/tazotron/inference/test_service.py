from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from tazotron.inference.model import build_binary_model
from tazotron.inference.schemas import ClassificationRequest
from tazotron.inference.service import InferenceFacade, RegisteredModel, RegisteredModelMetadata
from tazotron.inference.settings import InferenceSettings


class FakeTask:
    def __init__(self, task_id: str) -> None:
        self.id = task_id


class FakeClearmlClient:
    def __init__(self, checkpoint_path: Path, task_id: str = "task-123") -> None:
        self.checkpoint_path = checkpoint_path
        self.task = FakeTask(task_id)

    def get_task(self, **_: Any) -> FakeTask:
        return self.task

    def get_task_artifact_path(self, task: FakeTask, alias: str) -> Path:
        del alias
        return self.checkpoint_path


def _write_checkpoint(path: Path) -> Path:
    model = build_binary_model("resnet18", 2)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {"model_name": "resnet18"},
        "val_metrics": {"accuracy": 0.85, "loss": 0.34},
    }
    torch.save(checkpoint, path)
    return path


class TestInferenceFacade:
    @pytest.mark.fast
    def test_service_module_does_not_import_clearml_at_import_time(self) -> None:
        sys.modules.pop("tazotron.inference.service", None)

        module = importlib.import_module("tazotron.inference.service")

        assert "clearml" not in module.__dict__

    @pytest.mark.fast
    def test_load_model_metadata_from_clearml_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        checkpoint_path = _write_checkpoint(tmp_path / "checkpoint.pt")
        settings = InferenceSettings(model_name="resnet50", model_external_id="model-v1")
        facade = InferenceFacade(settings)
        clearml_client = FakeClearmlClient(checkpoint_path)

        monkeypatch.setattr(facade, "_get_clearml_client", lambda: clearml_client)

        model = facade._load_model_metadata()

        assert model.external_id == "model-v1"
        assert model.name == settings.model_display_name
        assert model.version == settings.clearml_task_name
        assert model.accuracy == pytest.approx(0.85)
        assert model.loss == pytest.approx(0.34)

    @pytest.mark.fast
    def test_list_models_uses_static_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = InferenceSettings(model_external_id="model-v1", model_display_name="Model V1")
        facade = InferenceFacade(settings)

        def _unexpected_load() -> RegisteredModelMetadata:
            raise AssertionError("metadata loading should not happen for /models")

        monkeypatch.setattr(facade, "_load_model_metadata", _unexpected_load)

        response = facade.list_models()

        assert [item.model_dump() for item in response] == [
            {
                "externalId": "model-v1",
                "name": "Model V1",
                "version": settings.clearml_task_name,
            }
        ]

    @pytest.mark.fast
    def test_get_metrics_avoids_building_classifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = InferenceSettings(model_external_id="model-v1")
        facade = InferenceFacade(settings)
        metadata = RegisteredModelMetadata(
            external_id="model-v1",
            name="Model V1",
            version="v1",
            accuracy=0.85,
            loss=0.34,
            checkpoint={"model_state_dict": {}},
            resolved_model_name="resnet18",
        )

        monkeypatch.setattr(facade, "_load_model_metadata", lambda: metadata)

        def _unexpected_build(_: RegisteredModelMetadata) -> RegisteredModel:
            raise AssertionError("classifier should not be built for /metrics")

        monkeypatch.setattr(facade, "_build_registered_model", _unexpected_build)

        response = facade.get_metrics("model-v1")

        assert response.model_dump() == {
            "externalId": "model-v1",
            "accuracy": 0.85,
            "loss": 0.34,
        }

    @pytest.mark.fast
    def test_facade_caches_registered_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = InferenceSettings(model_external_id="model-v1")
        facade = InferenceFacade(settings)
        calls = {"count": 0}

        metadata = RegisteredModelMetadata(
            external_id="model-v1",
            name="Fake",
            version="v1",
            accuracy=0.5,
            loss=0.7,
            checkpoint={"model_state_dict": {}},
            resolved_model_name="resnet18",
        )

        def _fake_build(_: RegisteredModelMetadata) -> Any:
            calls["count"] += 1

            class _Classifier:
                def predict(self, image_bytes: bytes) -> tuple[str, float]:
                    del image_bytes
                    return "negative", 0.1

            from tazotron.inference.service import RegisteredModel

            return RegisteredModel(
                external_id="model-v1",
                name="Fake",
                version="v1",
                accuracy=0.5,
                loss=0.7,
                classifier=_Classifier(),
            )

        monkeypatch.setattr(facade, "_load_model_metadata", lambda: metadata)
        monkeypatch.setattr(facade, "_build_registered_model", _fake_build)

        request = ClassificationRequest(modelExternalId="model-v1", imageBase64="ZGF0YQ==")
        facade.classify(request)
        facade.classify(request)

        assert calls["count"] == 1
