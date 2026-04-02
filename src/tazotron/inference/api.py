"""FastAPI application for model discovery and classification."""

from __future__ import annotations

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import Response

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
from tazotron.inference.service import InferenceFacade, get_inference_facade
from tazotron.inference.settings import get_settings

app = FastAPI(
    title="Tazotron Inference Service",
    version="0.1.0",
    description="External inference service compatible with the current backend contract.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/models", response_model=list[ModelResponse])
def list_models(
    inference_facade: InferenceFacade = Depends(get_inference_facade),
) -> list[ModelResponse]:
    """List available models."""
    return inference_facade.list_models()


@app.get("/models/{external_id}/metrics", response_model=ModelMetricsResponse)
def get_model_metrics(
    external_id: str,
    inference_facade: InferenceFacade = Depends(get_inference_facade),
) -> ModelMetricsResponse:
    """Return accuracy and loss for a model."""
    return inference_facade.get_metrics(external_id)


@app.post("/classify", response_model=ClassificationResponse)
def classify(
    request: ClassificationRequest,
    inference_facade: InferenceFacade = Depends(get_inference_facade),
) -> ClassificationResponse:
    """Classify an X-ray image from a base64 payload."""
    return inference_facade.classify(request)


@app.post("/ct-to-xray", response_class=Response)
def ct_to_xray(
    request: CtToXrayRequest,
    inference_facade: InferenceFacade = Depends(get_inference_facade),
) -> Response:
    """Render a synthetic X-ray image from an uploaded CT study."""
    xray_bytes = inference_facade.generate_xray_from_ct_request(request)
    return Response(content=xray_bytes, media_type="image/png")


@app.get("/training/architectures", response_model=list[TrainableArchitectureResponse])
def list_trainable_architectures(
    inference_facade: InferenceFacade = Depends(get_inference_facade),
) -> list[TrainableArchitectureResponse]:
    """List architectures available for admin-triggered training."""
    return inference_facade.list_trainable_architectures()


@app.post("/training/jobs", response_model=TrainingJobResponse)
def start_training_job(
    request: TrainingJobStartRequest,
    inference_facade: InferenceFacade = Depends(get_inference_facade),
) -> TrainingJobResponse:
    """Start an asynchronous training job."""
    return inference_facade.start_training_job(request)


@app.get("/training/jobs/current", response_model=TrainingJobResponse)
def get_current_training_job(
    inference_facade: InferenceFacade = Depends(get_inference_facade),
) -> TrainingJobResponse:
    """Return the currently active training job."""
    return inference_facade.get_current_training_job()


@app.get("/training/jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(
    job_id: str,
    inference_facade: InferenceFacade = Depends(get_inference_facade),
) -> TrainingJobResponse:
    """Return one training job by id."""
    return inference_facade.get_training_job(job_id)


def run() -> None:
    """Run the inference service with uvicorn."""
    settings = get_settings()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
