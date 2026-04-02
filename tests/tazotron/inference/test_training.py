from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from pathlib import Path
import time

import pytest
from fastapi import HTTPException
from PIL import Image

from tazotron.inference.model import build_binary_model
from tazotron.inference.schemas import ClassificationRequest, TrainingJobStartRequest
from tazotron.inference.service import InferenceFacade
from tazotron.inference.settings import InferenceSettings
from tazotron.inference.training import (
    TRAINABLE_ARCHITECTURES,
    TrainingArtifact,
    TrainingJobManager,
    load_local_models,
)


def _make_image_base64() -> str:
    image = Image.new("L", (8, 8), color=128)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("ascii")


def _wait_for_terminal_state(manager: TrainingJobManager, job_id: str) -> None:
    for _ in range(100):
        job = manager.get_job(job_id)
        if job.status in {"completed", "failed"}:
            return
        time.sleep(0.05)
    msg = f"Training job {job_id} did not reach a terminal state in time"
    raise AssertionError(msg)


def _build_artifact() -> TrainingArtifact:
    model = build_binary_model("resnet18", 2, loader_kind="timm")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": "resnet18",
            "loader_kind": "timm",
            "architecture_key": "resnet18",
            "dataset_name": "dataset-alpha",
            "learning_rate": 0.001,
            "epochs": 3,
            "batch_size": 2,
        },
        "val_metrics": {
            "accuracy": 0.75,
            "loss": 0.42,
        },
    }
    return TrainingArtifact(checkpoint=checkpoint, accuracy=0.75, loss=0.42)


class TestTrainingJobManager:
    @pytest.mark.fast
    def test_list_architectures_returns_expected_catalog(self) -> None:
        settings = InferenceSettings()
        manager = TrainingJobManager(settings)

        responses = manager.list_architectures()

        assert [item.key for item in responses] == list(TRAINABLE_ARCHITECTURES)

    @pytest.mark.fast
    def test_start_job_rejects_missing_radiologynet_weights(self, tmp_path: Path) -> None:
        settings = InferenceSettings(datasets_root_path=tmp_path)
        (tmp_path / "datasets" / "dataset-alpha").mkdir(parents=True)
        manager = TrainingJobManager(settings)

        with pytest.raises(HTTPException) as exc_info:
            manager.start_job(
                TrainingJobStartRequest(
                    architectureKey="radiologynet_resnet50_finetune",
                    datasetName="dataset-alpha",
                    learningRate=0.001,
                    epochs=3,
                    batchSize=2,
                )
            )

        assert exc_info.value.status_code == 400
        assert "Manual RadiologyNET weights were not found" in str(exc_info.value.detail)

    @pytest.mark.fast
    def test_successful_job_persists_local_model_and_classification_can_use_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        settings = InferenceSettings(
            datasets_root_path=tmp_path,
            model_external_id="clearml-model",
            model_device="cpu",
            image_size=8,
        )
        (tmp_path / "datasets" / "dataset-alpha").mkdir(parents=True)
        manager = TrainingJobManager(settings)

        monkeypatch.setattr(
            manager,
            "_render_training_dataset",
            lambda *_: None,
        )
        monkeypatch.setattr(
            manager,
            "_train_model",
            lambda *_: _build_artifact(),
        )

        started = manager.start_job(
            TrainingJobStartRequest(
                architectureKey="resnet18",
                datasetName="dataset-alpha",
                learningRate=0.001,
                epochs=3,
                batchSize=2,
            )
        )
        _wait_for_terminal_state(manager, started.jobId)
        finished_job = manager.get_job(started.jobId)

        assert finished_job.status == "completed"
        assert finished_job.modelExternalId is not None

        local_models = load_local_models(settings)
        assert finished_job.modelExternalId in local_models

        facade = InferenceFacade(settings)
        listed_model_ids = [model.externalId for model in facade.list_models()]
        assert "clearml-model" in listed_model_ids
        assert finished_job.modelExternalId in listed_model_ids

        result = facade.classify(
            ClassificationRequest(
                modelExternalId=finished_job.modelExternalId,
                imageBase64=_make_image_base64(),
            )
        )

        assert result.status == "success"
        assert result.diagnosis in {"positive", "negative"}
        assert result.probability is not None
