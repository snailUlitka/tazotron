"""Training helpers and in-memory orchestration for admin-triggered jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import threading
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, random_split

from tazotron.inference.model import (
    build_binary_model,
    ensure_three_channels,
    minmax_normalize,
)
from tazotron.inference.schemas import (
    TrainableArchitectureResponse,
    TrainingJobResponse,
    TrainingJobStartRequest,
)
from tazotron.inference.settings import InferenceSettings
from tazotron.xray_generation import render_xray_dataset_from_ct

RUNTIME_LOADER_KIND_TIMM = "timm"
RADIOLOGYNET_WEIGHTS_FILE_NAME = "radiologynet_resnet50.pth"
MODEL_FILE_SUFFIX = ".pt"
METADATA_FILE_SUFFIX = ".json"
RESERVED_ROOT_DIRECTORIES = {"datasets", "models", "weights"}
MODEL_ID_SANITIZER = re.compile(r"[^A-Za-z0-9._-]+")
KNOWN_PREFIXES = ("module.", "model.", "net.", "backbone.", "encoder.")
COMMON_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model_state", "model", "net", "backbone")


@dataclass(slots=True, frozen=True)
class TrainableArchitecture:
    key: str
    name: str
    description: str
    model_name: str
    loader_kind: str
    requires_manual_weights: bool = False

    def to_response(self) -> TrainableArchitectureResponse:
        return TrainableArchitectureResponse(
            key=self.key,
            name=self.name,
            description=self.description,
            requiresManualWeights=self.requires_manual_weights,
        )


@dataclass(slots=True, frozen=True)
class LocalModelRecord:
    external_id: str
    name: str
    version: str
    accuracy: float
    loss: float
    architecture_key: str
    dataset_name: str
    model_name: str
    loader_kind: str
    checkpoint_path: Path

    @staticmethod
    def from_metadata_file(metadata_path: Path) -> LocalModelRecord | None:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        checkpoint_file = payload.get("checkpointFile")
        if not isinstance(checkpoint_file, str):
            return None

        checkpoint_path = metadata_path.parent / checkpoint_file
        required_values = (
            payload.get("externalId"),
            payload.get("name"),
            payload.get("version"),
            payload.get("architectureKey"),
            payload.get("datasetName"),
            payload.get("modelName"),
            payload.get("loaderKind"),
        )
        if any(not isinstance(value, str) or not value for value in required_values):
            return None

        accuracy = payload.get("accuracy")
        loss = payload.get("loss")
        if not isinstance(accuracy, (int, float)) or not isinstance(loss, (int, float)):
            return None

        return LocalModelRecord(
            external_id=payload["externalId"],
            name=payload["name"],
            version=payload["version"],
            accuracy=float(accuracy),
            loss=float(loss),
            architecture_key=payload["architectureKey"],
            dataset_name=payload["datasetName"],
            model_name=payload["modelName"],
            loader_kind=payload["loaderKind"],
            checkpoint_path=checkpoint_path,
        )


@dataclass(slots=True, frozen=True)
class TrainingArtifact:
    checkpoint: dict[str, Any]
    accuracy: float
    loss: float


@dataclass(slots=True)
class TrainingJobState:
    job_id: str
    status: str
    dataset_name: str
    architecture_key: str
    learning_rate: float
    epochs: int
    batch_size: int
    current_epoch: int | None
    total_epochs: int | None
    progress_message: str | None
    model_external_id: str | None
    error_message: str | None
    started_at: datetime
    finished_at: datetime | None

    def to_response(self) -> TrainingJobResponse:
        return TrainingJobResponse(
            jobId=self.job_id,
            status=self.status,
            datasetName=self.dataset_name,
            architectureKey=self.architecture_key,
            learningRate=self.learning_rate,
            epochs=self.epochs,
            batchSize=self.batch_size,
            currentEpoch=self.current_epoch,
            totalEpochs=self.total_epochs,
            progressMessage=self.progress_message,
            modelExternalId=self.model_external_id,
            errorMessage=self.error_message,
            startedAt=self.started_at,
            finishedAt=self.finished_at,
        )


TRAINABLE_ARCHITECTURES = {
    architecture.key: architecture
    for architecture in (
        TrainableArchitecture(
            key="resnet18",
            name="ResNet18",
            description="Compact ResNet18 classifier trained from scratch.",
            model_name="resnet18",
            loader_kind=RUNTIME_LOADER_KIND_TIMM,
        ),
        TrainableArchitecture(
            key="resnet50",
            name="ResNet50",
            description="Larger ResNet50 classifier trained from scratch.",
            model_name="resnet50",
            loader_kind=RUNTIME_LOADER_KIND_TIMM,
        ),
        TrainableArchitecture(
            key="radiologynet_resnet50_finetune",
            name="RadiologyNET ResNet50 finetune",
            description="ResNet50 initialized from manual RadiologyNET backbone weights.",
            model_name="resnet50",
            loader_kind=RUNTIME_LOADER_KIND_TIMM,
            requires_manual_weights=True,
        ),
    )
}


class RenderedXrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load rendered `.pt` X-rays for binary classification."""

    def __init__(self, root_path: Path, settings: InferenceSettings) -> None:
        with_necro_dir = root_path / "with_necro"
        without_necro_dir = root_path / "without_necro"
        self.samples: list[tuple[Path, int]] = []
        self.samples.extend((path, 1) for path in sorted(with_necro_dir.glob("*.pt")))
        self.samples.extend((path, 0) for path in sorted(without_necro_dir.glob("*.pt")))
        if not self.samples:
            msg = f"No rendered X-rays found in {root_path}"
            raise ValueError(msg)
        self.settings = settings

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        tensor = torch.load(path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            msg = f"Expected a tensor in {path}"
            raise TypeError(msg)

        image = tensor.to(torch.float32)
        if image.ndim == 4:
            image = image[0, 0]
        elif image.ndim == 3 and image.shape[0] == 1:
            image = image[0]
        image = minmax_normalize(image)
        image = ensure_three_channels(image)
        image = functional.interpolate(
            image.unsqueeze(0),
            size=(self.settings.image_size, self.settings.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mean = torch.tensor(self.settings.normalize_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.settings.normalize_std, dtype=torch.float32).view(3, 1, 1)
        image = (image - mean) / std
        return image, torch.tensor(label, dtype=torch.long)


class TrainingJobManager:
    """Owns the single active training job and exposes polling-friendly snapshots."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self._jobs: dict[str, TrainingJobState] = {}
        self._current_job_id: str | None = None
        self._lock = threading.Lock()

    def list_architectures(self) -> list[TrainableArchitectureResponse]:
        return [architecture.to_response() for architecture in TRAINABLE_ARCHITECTURES.values()]

    def start_job(self, request: TrainingJobStartRequest) -> TrainingJobResponse:
        architecture = self._get_architecture(request.architectureKey)
        self._resolve_dataset_path(request.datasetName)
        self._validate_architecture_dependencies(architecture)
        started_at = datetime.now(UTC)
        job_id = uuid4().hex

        with self._lock:
            if self._current_job_id is not None:
                current_job = self._jobs[self._current_job_id]
                if current_job.status in {"queued", "running"}:
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Training job is already running")

            job = TrainingJobState(
                job_id=job_id,
                status="queued",
                dataset_name=request.datasetName,
                architecture_key=request.architectureKey,
                learning_rate=request.learningRate,
                epochs=request.epochs,
                batch_size=request.batchSize,
                current_epoch=0,
                total_epochs=request.epochs,
                progress_message="Training job queued",
                model_external_id=None,
                error_message=None,
                started_at=started_at,
                finished_at=None,
            )
            self._jobs[job_id] = job
            self._current_job_id = job_id

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, architecture, request),
            name=f"training-job-{job_id}",
            daemon=True,
        )
        thread.start()
        return job.to_response()

    def get_current_job(self) -> TrainingJobResponse:
        with self._lock:
            if self._current_job_id is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active training job")
            return self._jobs[self._current_job_id].to_response()

    def get_job(self, job_id: str) -> TrainingJobResponse:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job '{job_id}' was not found")
            return job.to_response()

    def _run_job(self, job_id: str, architecture: TrainableArchitecture, request: TrainingJobStartRequest) -> None:
        try:
            dataset_path = self._resolve_dataset_path(request.datasetName)
            self._update_job(
                job_id,
                status="running",
                progress_message="Rendering synthetic X-rays from the selected dataset",
                current_epoch=0,
                total_epochs=request.epochs,
            )

            with TemporaryDirectory(prefix=f"tazotron-training-{job_id}-") as temp_dir:
                rendered_dataset_path = Path(temp_dir) / "rendered"
                self._render_training_dataset(dataset_path, rendered_dataset_path)
                self._update_job(
                    job_id,
                    progress_message="Synthetic dataset is ready. Starting model training",
                )

                artifact = self._train_model(
                    architecture,
                    request,
                    rendered_dataset_path,
                    job_id,
                )
                local_model = self._save_trained_model(artifact, architecture, request, job_id)
                self._update_job(
                    job_id,
                    status="completed",
                    current_epoch=request.epochs,
                    total_epochs=request.epochs,
                    progress_message="Training finished successfully",
                    model_external_id=local_model.external_id,
                    finished_at=datetime.now(UTC),
                )
        except HTTPException as error:
            self._mark_failed(job_id, str(error.detail))
        except Exception as error:  # noqa: BLE001
            self._mark_failed(job_id, str(error))
        finally:
            with self._lock:
                if self._current_job_id == job_id:
                    self._current_job_id = None

    def _render_training_dataset(self, dataset_path: Path, rendered_dataset_path: Path) -> None:
        render_xray_dataset_from_ct(
            dataset_path,
            rendered_dataset_path,
            framing_mode="autopose",
            device=_resolve_device(self.settings.model_device),
            overwrite_existing=True,
        )

    def _train_model(
        self,
        architecture: TrainableArchitecture,
        request: TrainingJobStartRequest,
        rendered_dataset_path: Path,
        job_id: str,
    ) -> TrainingArtifact:
        dataset = RenderedXrayDataset(rendered_dataset_path, self.settings)
        if len(dataset) < 2:
            msg = "Rendered training dataset must contain at least two samples"
            raise ValueError(msg)

        validation_size = max(1, len(dataset) // 5)
        if validation_size >= len(dataset):
            validation_size = 1
        train_size = len(dataset) - validation_size
        if train_size <= 0:
            msg = "Rendered training dataset is too small to create a validation split"
            raise ValueError(msg)

        generator = torch.Generator().manual_seed(42)
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=request.batchSize, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=request.batchSize, shuffle=False)

        device = torch.device(_resolve_device(self.settings.model_device))
        model = build_binary_model(
            architecture.model_name,
            2,
            loader_kind=architecture.loader_kind,
        ).to(device)
        if architecture.requires_manual_weights:
            _load_radiologynet_weights(model, self.settings.weights_root_path / RADIOLOGYNET_WEIGHTS_FILE_NAME)

        optimizer = torch.optim.Adam(model.parameters(), lr=request.learningRate)
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        best_accuracy = 0.0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(1, request.epochs + 1):
            model.train()
            running_loss = 0.0
            seen_samples = 0
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                batch_size = inputs.shape[0]
                running_loss += float(loss.item()) * batch_size
                seen_samples += batch_size

            val_loss, val_accuracy = _evaluate_model(model, validation_loader, device, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_accuracy = val_accuracy
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            train_loss = running_loss / max(seen_samples, 1)
            self._update_job(
                job_id,
                current_epoch=epoch,
                total_epochs=request.epochs,
                progress_message=(
                    f"Epoch {epoch}/{request.epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
                ),
            )

        if best_state is None:
            msg = "Training finished without a best checkpoint"
            raise RuntimeError(msg)

        return TrainingArtifact(
            checkpoint={
                "model_state_dict": best_state,
                "config": {
                    "model_name": architecture.model_name,
                    "loader_kind": architecture.loader_kind,
                    "architecture_key": architecture.key,
                    "dataset_name": request.datasetName,
                    "learning_rate": request.learningRate,
                    "epochs": request.epochs,
                    "batch_size": request.batchSize,
                },
                "val_metrics": {
                    "accuracy": best_accuracy,
                    "loss": best_val_loss,
                },
            },
            accuracy=best_accuracy,
            loss=best_val_loss,
        )

    def _save_trained_model(
        self,
        artifact: TrainingArtifact,
        architecture: TrainableArchitecture,
        request: TrainingJobStartRequest,
        job_id: str,
    ) -> LocalModelRecord:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        dataset_slug = _slugify(request.datasetName)
        external_id = f"local-{architecture.key}-{dataset_slug}-{timestamp}"
        version = timestamp
        checkpoint_file_name = f"{external_id}{MODEL_FILE_SUFFIX}"
        metadata_file_name = f"{external_id}{METADATA_FILE_SUFFIX}"
        models_root_path = self.settings.models_root_path
        models_root_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = models_root_path / checkpoint_file_name
        metadata_path = models_root_path / metadata_file_name

        torch.save(artifact.checkpoint, checkpoint_path)
        metadata = {
            "externalId": external_id,
            "name": f"{architecture.name} ({request.datasetName})",
            "version": version,
            "accuracy": artifact.accuracy,
            "loss": artifact.loss,
            "architectureKey": architecture.key,
            "datasetName": request.datasetName,
            "modelName": architecture.model_name,
            "loaderKind": architecture.loader_kind,
            "checkpointFile": checkpoint_file_name,
            "jobId": job_id,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
        return LocalModelRecord.from_metadata_file(metadata_path) or LocalModelRecord(
            external_id=external_id,
            name=metadata["name"],
            version=version,
            accuracy=artifact.accuracy,
            loss=artifact.loss,
            architecture_key=architecture.key,
            dataset_name=request.datasetName,
            model_name=architecture.model_name,
            loader_kind=architecture.loader_kind,
            checkpoint_path=checkpoint_path,
        )

    def _resolve_dataset_path(self, dataset_name: str) -> Path:
        preferred_path = self.settings.datasets_catalog_path / dataset_name
        if preferred_path.is_dir():
            return preferred_path

        legacy_path = self.settings.data_root_path / dataset_name
        if legacy_path.is_dir() and legacy_path.name not in RESERVED_ROOT_DIRECTORIES:
            return legacy_path

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_name}' was not found",
        )

    def _get_architecture(self, architecture_key: str) -> TrainableArchitecture:
        architecture = TRAINABLE_ARCHITECTURES.get(architecture_key)
        if architecture is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Architecture '{architecture_key}' was not found",
        )
        return architecture

    def _validate_architecture_dependencies(self, architecture: TrainableArchitecture) -> None:
        if architecture.requires_manual_weights and not (self.settings.weights_root_path / RADIOLOGYNET_WEIGHTS_FILE_NAME).is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Manual RadiologyNET weights were not found at "
                    f"{self.settings.weights_root_path / RADIOLOGYNET_WEIGHTS_FILE_NAME}"
                ),
            )

    def _mark_failed(self, job_id: str, error_message: str) -> None:
        self._update_job(
            job_id,
            status="failed",
            error_message=error_message or "Training job failed",
            progress_message="Training job failed",
            finished_at=datetime.now(UTC),
        )

    def _update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        current_epoch: int | None = None,
        total_epochs: int | None = None,
        progress_message: str | None = None,
        model_external_id: str | None = None,
        error_message: str | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                job.status = status
            if current_epoch is not None:
                job.current_epoch = current_epoch
            if total_epochs is not None:
                job.total_epochs = total_epochs
            if progress_message is not None:
                job.progress_message = progress_message
            if model_external_id is not None:
                job.model_external_id = model_external_id
            if error_message is not None:
                job.error_message = error_message
            if finished_at is not None:
                job.finished_at = finished_at


def load_local_models(settings: InferenceSettings) -> dict[str, LocalModelRecord]:
    """Load persisted local model metadata from `.data/models`."""
    models_root_path = settings.models_root_path
    if not models_root_path.is_dir():
        return {}

    records: dict[str, LocalModelRecord] = {}
    for metadata_path in sorted(models_root_path.glob(f"*{METADATA_FILE_SUFFIX}")):
        record = LocalModelRecord.from_metadata_file(metadata_path)
        if record is None:
            continue
        records[record.external_id] = record
    return records


def _evaluate_model(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = inputs.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((outputs.argmax(dim=1) == targets).sum().item())
            total_samples += batch_size

    mean_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return mean_loss, accuracy


def _load_radiologynet_weights(model: nn.Module, weights_path: Path) -> None:
    if not weights_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Manual RadiologyNET weights were not found at {weights_path}",
        )

    checkpoint = torch.load(weights_path, map_location="cpu")
    candidates = _extract_candidate_state_dicts(checkpoint)
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Manual RadiologyNET weights do not contain a usable state_dict",
        )

    best_candidate: dict[str, torch.Tensor] | None = None
    best_loaded_key_count = -1
    for candidate in candidates:
        filtered = _filter_state_dict_for_model(model, candidate)
        if len(filtered) > best_loaded_key_count:
            best_candidate = filtered
            best_loaded_key_count = len(filtered)

    if not best_candidate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Manual RadiologyNET weights are incompatible with the selected ResNet50 backbone",
        )

    model.load_state_dict(best_candidate, strict=False)


def _extract_candidate_state_dicts(checkpoint: object) -> list[dict[str, torch.Tensor]]:
    candidates: list[dict[str, torch.Tensor]] = []
    if _looks_like_state_dict(checkpoint):
        candidates.append(_normalize_state_dict_keys(checkpoint))
    if isinstance(checkpoint, dict):
        for key in COMMON_STATE_DICT_KEYS:
            value = checkpoint.get(key)
            if _looks_like_state_dict(value):
                candidates.append(_normalize_state_dict_keys(value))
    return candidates


def _looks_like_state_dict(value: object) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    )


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key
        changed = True
        while changed:
            changed = False
            for prefix in KNOWN_PREFIXES:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix):]
                    changed = True
        normalized[normalized_key] = value
    return normalized


def _filter_state_dict_for_model(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }


def _resolve_device(configured_device: str) -> str:
    if configured_device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return configured_device


def _slugify(value: str) -> str:
    cleaned = MODEL_ID_SANITIZER.sub("-", value.strip())
    return cleaned.strip("-") or "dataset"
