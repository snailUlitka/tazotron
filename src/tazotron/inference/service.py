"""Service layer for the FastAPI inference application."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from tazotron.inference.model import (
    BinaryImageClassifier,
    build_binary_model,
    extract_metrics,
    extract_model_state_dict,
    load_checkpoint,
    resolve_model_name,
)
from tazotron.inference.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    ModelMetricsResponse,
    ModelResponse,
)
from tazotron.inference.settings import InferenceSettings, get_settings


@dataclass(slots=True, frozen=True)
class RegisteredModel:
    """A single model exposed by the inference service."""

    external_id: str
    name: str
    version: str
    accuracy: float
    loss: float
    classifier: BinaryImageClassifier

    def to_model_response(self) -> ModelResponse:
        return ModelResponse(
            externalId=self.external_id,
            name=self.name,
            version=self.version,
        )

    def to_metrics_response(self) -> ModelMetricsResponse:
        return ModelMetricsResponse(
            externalId=self.external_id,
            accuracy=self.accuracy,
            loss=self.loss,
        )


@dataclass(slots=True, frozen=True)
class RegisteredModelMetadata:
    """Checkpoint-backed metadata that does not require a live classifier."""

    external_id: str
    name: str
    version: str
    accuracy: float
    loss: float
    checkpoint: dict[str, Any]
    resolved_model_name: str

    def to_model_response(self) -> ModelResponse:
        return ModelResponse(
            externalId=self.external_id,
            name=self.name,
            version=self.version,
        )

    def to_metrics_response(self) -> ModelMetricsResponse:
        return ModelMetricsResponse(
            externalId=self.external_id,
            accuracy=self.accuracy,
            loss=self.loss,
        )


class InferenceFacade:
    """Facade that keeps the backend-facing API stable."""

    def __init__(self, settings: InferenceSettings) -> None:
        self.settings = settings
        self._configured_model = ModelResponse(
            externalId=settings.model_external_id,
            name=settings.model_display_name,
            version=self._resolve_version(),
        )
        self._model_metadata: RegisteredModelMetadata | None = None
        self._registered_model: RegisteredModel | None = None

    def list_models(self) -> list[ModelResponse]:
        return [self._configured_model]

    def get_metrics(self, external_id: str) -> ModelMetricsResponse:
        return self._get_model_metadata(external_id).to_metrics_response()

    def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        model = self._get_registered_model(request.modelExternalId)
        image_bytes = self._decode_image(request.imageBase64)
        try:
            diagnosis, probability = model.classifier.predict(image_bytes)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error
        return ClassificationResponse(
            status="success",
            diagnosis=diagnosis,
            probability=probability,
            errorMessage=None,
        )

    def _get_registered_model(self, external_id: str) -> RegisteredModel:
        self._ensure_known_model(external_id)
        if self._registered_model is None:
            metadata = self._get_model_metadata(external_id)
            self._registered_model = self._build_registered_model(metadata)
        return self._registered_model

    def _get_model_metadata(self, external_id: str) -> RegisteredModelMetadata:
        self._ensure_known_model(external_id)
        if self._model_metadata is None:
            self._model_metadata = self._load_model_metadata()
        return self._model_metadata

    def _load_model_metadata(self) -> RegisteredModelMetadata:
        try:
            clearml_client = self._get_clearml_client()
            task = clearml_client.get_task(
                project_name=self.settings.clearml_project_name,
                task_name=self.settings.clearml_task_name,
            )
            checkpoint_path = clearml_client.get_task_artifact_path(
                task,
                alias=self.settings.clearml_artifact_name,
            )
            checkpoint = load_checkpoint(Path(checkpoint_path))
            accuracy, loss = extract_metrics(checkpoint)
            version = self._resolve_version()
        except ImportError as error:
            raise self._service_unavailable("clearml package is not available") from error
        except FileNotFoundError as error:
            raise self._service_unavailable(str(error)) from error
        except (KeyError, RuntimeError, ValueError) as error:
            raise self._service_unavailable(str(error)) from error

        return RegisteredModelMetadata(
            external_id=self.settings.model_external_id,
            name=self.settings.model_display_name,
            version=version,
            accuracy=accuracy,
            loss=loss,
            checkpoint=checkpoint,
            resolved_model_name=resolve_model_name(self.settings, checkpoint),
        )

    def _build_registered_model(self, metadata: RegisteredModelMetadata) -> RegisteredModel:
        try:
            model = build_binary_model(metadata.resolved_model_name, self.settings.model_num_classes)
            state_dict = extract_model_state_dict(metadata.checkpoint)
            model.load_state_dict(state_dict)
        except (KeyError, RuntimeError, ValueError) as error:
            raise self._service_unavailable(str(error)) from error

        classifier = BinaryImageClassifier(model=model, settings=self.settings)
        return RegisteredModel(
            external_id=metadata.external_id,
            name=metadata.name,
            version=metadata.version,
            accuracy=metadata.accuracy,
            loss=metadata.loss,
            classifier=classifier,
        )

    def _decode_image(self, image_base64: str) -> bytes:
        try:
            image_bytes = b64decode(image_base64, validate=True)
        except BinasciiError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="imageBase64 must contain a valid base64 payload",
            ) from error
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="imageBase64 must not decode to an empty payload",
            )
        return image_bytes

    def _service_unavailable(self, message: str) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=message,
        )

    def _ensure_known_model(self, external_id: str) -> None:
        if external_id != self.settings.model_external_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{external_id}' was not found",
            )

    def _get_clearml_client(self) -> Any:
        from tazotron.integrations import clearml

        return clearml

    def _resolve_version(self) -> str:
        return self.settings.model_version or self.settings.clearml_task_name


@lru_cache(maxsize=1)
def get_inference_facade() -> InferenceFacade:
    """Return the cached service facade."""
    return InferenceFacade(get_settings())


def reset_inference_state() -> None:
    """Clear settings and service caches, mostly for tests."""
    get_settings.cache_clear()
    get_inference_facade.cache_clear()
