from __future__ import annotations

import importlib
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import torch
import torchio as tio
from fastapi import HTTPException
from PIL import Image

from tazotron.inference.model import build_binary_model
from tazotron.inference.schemas import ClassificationRequest
from tazotron.inference.service import CtSegmentationInput, InferenceFacade, RegisteredModel, RegisteredModelMetadata
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


class _DummyDRR:
    def __init__(self, subject: tio.Subject, **_: object) -> None:
        self.subject = subject

    def to(self, device: torch.device | str) -> _DummyDRR:
        del device
        return self

    def __call__(self, rotations: torch.Tensor, translations: torch.Tensor, **_: object) -> torch.Tensor:
        del rotations, translations
        volume = self.subject["volume"].data
        projection = volume[0].sum(dim=0, keepdim=True).unsqueeze(0)
        if float(projection.max().item() - projection.min().item()) < 1e-6:
            projection = projection.clone()
            projection[..., 0, 0] = 1.0
        return projection


def _serialize_image(path: Path, tensor: torch.Tensor, *, label: bool = False) -> bytes:
    image_cls = tio.LabelMap if label else tio.ScalarImage
    image_cls(tensor=tensor, affine=torch.eye(4)).save(path)
    return path.read_bytes()


def _make_ct_to_xray_payload(tmp_path: Path) -> tuple[bytes, list[CtSegmentationInput]]:
    ct_tensor = torch.full((1, 8, 8, 8), fill_value=100.0, dtype=torch.float32)
    left_mask = torch.zeros_like(ct_tensor, dtype=torch.int16)
    right_mask = torch.zeros_like(ct_tensor, dtype=torch.int16)
    left_mask[0, 1:6, 1:6, 1:6] = 1
    right_mask[0, 2:7, 2:7, 2:7] = 1

    ct_bytes = _serialize_image(tmp_path / "scan.nii.gz", ct_tensor)
    segmentations = [
        CtSegmentationInput(
            type="femur_left",
            file_name="femur-left.nrrd",
            file_bytes=_serialize_image(tmp_path / "femur-left.nrrd", left_mask, label=True),
        ),
        CtSegmentationInput(
            type="femur_right",
            file_name="femur-right.nrrd",
            file_bytes=_serialize_image(tmp_path / "femur-right.nrrd", right_mask, label=True),
        ),
    ]
    return ct_bytes, segmentations


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

        model = facade._load_configured_model_metadata()

        assert model.external_id == "model-v1"
        assert model.name == settings.model_display_name
        assert model.version == settings.clearml_task_name
        assert model.accuracy == pytest.approx(0.85)
        assert model.loss == pytest.approx(0.34)
        assert model.checkpoint_path == checkpoint_path

    @pytest.mark.fast
    def test_list_models_uses_static_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = InferenceSettings(model_external_id="model-v1", model_display_name="Model V1")
        facade = InferenceFacade(settings)

        def _unexpected_load() -> RegisteredModelMetadata:
            raise AssertionError("metadata loading should not happen for /models")

        monkeypatch.setattr(facade, "_load_configured_model_metadata", _unexpected_load)

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
        checkpoint_path = Path("checkpoint.pt")
        metadata = RegisteredModelMetadata(
            external_id="model-v1",
            name="Model V1",
            version="v1",
            accuracy=0.85,
            loss=0.34,
            checkpoint_path=checkpoint_path,
            resolved_model_name="resnet18",
            loader_kind="timm",
        )

        monkeypatch.setattr(facade, "_load_configured_model_metadata", lambda: metadata)

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
        checkpoint_path = Path("checkpoint.pt")

        metadata = RegisteredModelMetadata(
            external_id="model-v1",
            name="Fake",
            version="v1",
            accuracy=0.5,
            loss=0.7,
            checkpoint_path=checkpoint_path,
            resolved_model_name="resnet18",
            loader_kind="timm",
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

        monkeypatch.setattr(facade, "_load_configured_model_metadata", lambda: metadata)
        monkeypatch.setattr(facade, "_build_registered_model", _fake_build)

        request = ClassificationRequest(modelExternalId="model-v1", imageBase64="ZGF0YQ==")
        facade.classify(request)
        facade.classify(request)

        assert calls["count"] == 1

    @pytest.mark.fast
    def test_generate_xray_from_ct_returns_png(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        settings = InferenceSettings(model_external_id="model-v1", model_device="cpu")
        facade = InferenceFacade(settings)
        ct_bytes, segmentations = _make_ct_to_xray_payload(tmp_path)

        monkeypatch.setattr("tazotron.datasets.transforms.xray.DRR", _DummyDRR)

        xray_bytes = facade.generate_xray_from_ct(ct_bytes, segmentations)

        decoded = Image.open(BytesIO(xray_bytes))
        assert decoded.format == "PNG"
        assert decoded.size == (8, 8)

    @pytest.mark.fast
    def test_generate_xray_from_ct_rejects_missing_required_segmentations(self, tmp_path: Path) -> None:
        settings = InferenceSettings(model_external_id="model-v1", model_device="cpu")
        facade = InferenceFacade(settings)
        ct_bytes, segmentations = _make_ct_to_xray_payload(tmp_path)

        with pytest.raises(HTTPException) as exc_info:
            facade.generate_xray_from_ct(ct_bytes, segmentations[:1])

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "CT -> Xray requires segmentations: femur_right"
