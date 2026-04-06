"""Service layer for the FastAPI inference application."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import HTTPException, status
from PIL import Image
import torch

from tazotron.datasets.ct import CTDataset
from tazotron.datasets.transforms.femoral_head import AddFemoralHeadMasks
from tazotron.datasets.transforms.xray import RenderDRR
from tazotron.inference.model import (
    BinaryImageClassifier,
    build_binary_model,
    extract_metrics,
    extract_model_state_dict,
    load_checkpoint,
    resolve_loader_kind,
    resolve_model_name,
)
from tazotron.inference.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    CtToXrayRequest,
    ModelMetricsResponse,
    ModelResponse,
    TrainableArchitectureResponse,
    TrainingJobResponse,
    TrainingJobStartRequest,
)
from tazotron.inference.settings import InferenceSettings, get_settings
from tazotron.inference.training import (
    LocalModelRecord,
    TrainingJobManager,
    load_local_models,
)
from tazotron.xray_generation import AUTOPOSE_MODE, apply_framing, squeeze_xray_tensor

TEMP_CT_CASE_ID = "case-001"
TEMP_CT_FILE_NAME = "ct.nii.gz"
SEGMENTATION_FILE_NAMES = {
    "femur_left": "femur_left.nii.gz.seg.nrrd",
    "femur_right": "femur_right.nii.gz.seg.nrrd",
}
SEGMENTATION_TYPE_ALIASES = {
    "femur_left": "femur_left",
    "label_femur_left": "femur_left",
    "femur_right": "femur_right",
    "label_femur_right": "femur_right",
}


@dataclass(slots=True, frozen=True)
class CtSegmentationInput:
    """A CT segmentation payload accepted by the inference HTTP API."""

    type: str
    file_name: str
    file_bytes: bytes


@dataclass(slots=True, frozen=True)
class RegisteredModel:
    """A single model exposed by the inference service."""

    external_id: str
    name: str
    version: str
    accuracy: float
    loss: float
    classifier: BinaryImageClassifier


@dataclass(slots=True, frozen=True)
class RegisteredModelMetadata:
    """Model metadata that does not require a live classifier."""

    external_id: str
    name: str
    version: str
    accuracy: float
    loss: float
    checkpoint_path: Path
    resolved_model_name: str
    loader_kind: str

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
        self._configured_model_metadata: RegisteredModelMetadata | None = None
        self._registered_models: dict[str, RegisteredModel] = {}
        self.training_jobs = TrainingJobManager(settings)

    def list_models(self) -> list[ModelResponse]:
        models = [self._configured_model]
        local_models = load_local_models(self.settings)
        models.extend(
            ModelResponse(
                externalId=record.external_id,
                name=record.name,
                version=record.version,
            )
            for record in local_models.values()
        )
        return models

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

    def list_trainable_architectures(self) -> list[TrainableArchitectureResponse]:
        return self.training_jobs.list_architectures()

    def start_training_job(self, request: TrainingJobStartRequest) -> TrainingJobResponse:
        return self.training_jobs.start_job(request)

    def get_current_training_job(self) -> TrainingJobResponse:
        return self.training_jobs.get_current_job()

    def get_training_job(self, job_id: str) -> TrainingJobResponse:
        return self.training_jobs.get_job(job_id)

    def generate_xray_from_ct_request(self, request: CtToXrayRequest) -> bytes:
        ct_bytes = self._decode_base64_payload(request.ctFileBase64, "ctFileBase64")
        segmentations = [
            CtSegmentationInput(
                type=segmentation.type,
                file_name="",
                file_bytes=self._decode_base64_payload(
                    segmentation.fileBase64,
                    f"segmentations[{index}].fileBase64",
                ),
            )
            for index, segmentation in enumerate(request.segmentations)
        ]
        return self.generate_xray_from_ct(ct_bytes, segmentations)

    def generate_xray_from_ct(self, ct_bytes: bytes, segmentations: list[CtSegmentationInput]) -> bytes:
        if not ct_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ctFile must not be empty",
            )

        normalized_segmentations = self._normalize_ct_segmentations(segmentations)
        try:
            with TemporaryDirectory(prefix="tazotron-ct-to-xray-") as temp_dir:
                case_dir = Path(temp_dir) / TEMP_CT_CASE_ID
                case_dir.mkdir(parents=True, exist_ok=True)
                (case_dir / TEMP_CT_FILE_NAME).write_bytes(ct_bytes)
                for segmentation_type, segmentation in normalized_segmentations.items():
                    (case_dir / SEGMENTATION_FILE_NAMES[segmentation_type]).write_bytes(segmentation.file_bytes)

                femoral_head_masks = AddFemoralHeadMasks()
                render = RenderDRR({"device": self.settings.model_device})

                def pipeline(subject):
                    subject = femoral_head_masks(subject)
                    subject = apply_framing(subject, framing_mode=AUTOPOSE_MODE)
                    return render(subject)

                dataset = CTDataset(Path(temp_dir), transform=pipeline)
                subject = dataset[0]
                return self._encode_png(subject["xray"])
        except HTTPException:
            raise
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(error),
            ) from error

    def _get_registered_model(self, external_id: str) -> RegisteredModel:
        cached_model = self._registered_models.get(external_id)
        if cached_model is not None:
            return cached_model

        metadata = self._get_model_metadata(external_id)
        registered_model = self._build_registered_model(metadata)
        self._registered_models[external_id] = registered_model
        return registered_model

    def _get_model_metadata(self, external_id: str) -> RegisteredModelMetadata:
        if external_id == self.settings.model_external_id:
            if self._configured_model_metadata is None:
                self._configured_model_metadata = self._load_configured_model_metadata()
            return self._configured_model_metadata

        record = self._get_local_model_record(external_id)
        return RegisteredModelMetadata(
            external_id=record.external_id,
            name=record.name,
            version=record.version,
            accuracy=record.accuracy,
            loss=record.loss,
            checkpoint_path=record.checkpoint_path,
            resolved_model_name=record.model_name,
            loader_kind=record.loader_kind,
        )

    def _load_configured_model_metadata(self) -> RegisteredModelMetadata:
        try:
            clearml_client = self._get_clearml_client()
            task = clearml_client.get_task(
                project_name=self.settings.clearml_project_name,
                task_name=self.settings.clearml_task_name,
                tags=["radiologynet"],
            )
            checkpoint_path = clearml_client.get_task_artifact_path(
                task,
                alias=self.settings.clearml_artifact_name,
            )
            checkpoint = load_checkpoint(Path(checkpoint_path))
            accuracy, loss = extract_metrics(checkpoint)
            version = self._resolve_version()
            resolved_model_name = resolve_model_name(self.settings, checkpoint)
            loader_kind = resolve_loader_kind(checkpoint)
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
            checkpoint_path=Path(checkpoint_path),
            resolved_model_name=resolved_model_name,
            loader_kind=loader_kind,
        )

    def _build_registered_model(self, metadata: RegisteredModelMetadata) -> RegisteredModel:
        try:
            checkpoint = load_checkpoint(metadata.checkpoint_path)
            model = build_binary_model(
                metadata.resolved_model_name,
                self.settings.model_num_classes,
                loader_kind=metadata.loader_kind,
            )
            state_dict = extract_model_state_dict(checkpoint)
            model.load_state_dict(state_dict)
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
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

    def _get_local_model_record(self, external_id: str) -> LocalModelRecord:
        record = load_local_models(self.settings).get(external_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{external_id}' was not found",
            )
        return record

    def _decode_image(self, image_base64: str) -> bytes:
        return self._decode_base64_payload(image_base64, "imageBase64")

    def _decode_base64_payload(self, payload: str, field_name: str) -> bytes:
        try:
            decoded_bytes = b64decode(payload, validate=True)
        except BinasciiError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must contain a valid base64 payload",
            ) from error
        if not decoded_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name} must not decode to an empty payload",
            )
        return decoded_bytes

    def _normalize_ct_segmentations(
        self,
        segmentations: list[CtSegmentationInput],
    ) -> dict[str, CtSegmentationInput]:
        normalized: dict[str, CtSegmentationInput] = {}
        for segmentation in segmentations:
            raw_type = segmentation.type.strip().lower()
            if raw_type not in SEGMENTATION_TYPE_ALIASES:
                continue
            if not segmentation.file_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Segmentation '{segmentation.type}' must not be empty",
                )

            normalized_type = SEGMENTATION_TYPE_ALIASES[raw_type]
            if normalized_type in normalized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Duplicate segmentation type '{segmentation.type}'",
                )
            normalized[normalized_type] = segmentation

        missing_types = [
            segmentation_type for segmentation_type in SEGMENTATION_FILE_NAMES if segmentation_type not in normalized
        ]
        if missing_types:
            missing = ", ".join(missing_types)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CT -> Xray requires segmentations: {missing}",
            )
        return normalized

    def _encode_png(self, tensor) -> bytes:
        image = torch.nan_to_num(squeeze_xray_tensor(tensor))
        image -= image.min()
        max_value = float(image.max().item())
        if max_value > 0.0:
            image = image / max_value

        image_bytes = (image * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
        buffer = BytesIO()
        Image.fromarray(image_bytes).save(buffer, format="PNG")
        return buffer.getvalue()

    def _service_unavailable(self, message: str) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=message,
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
