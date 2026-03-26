"""FastAPI application for model discovery and classification."""

from __future__ import annotations

import uvicorn
from fastapi import Depends, FastAPI

from tazotron.inference.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    ModelMetricsResponse,
    ModelResponse,
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


def run() -> None:
    """Run the inference service with uvicorn."""
    settings = get_settings()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
