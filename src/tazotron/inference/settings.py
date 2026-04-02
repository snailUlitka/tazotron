"""Settings for the inference HTTP service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    """Configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file="configs/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_external_id: str = "radiologynet-binary-necrosis-autopose"
    model_display_name: str = "RadiologyNET Binary Necrosis Classifier"
    model_version: str | None = None
    model_name: str = "resnet50"
    model_device: str = "cpu"
    model_num_classes: int = 2
    classification_threshold: float = 0.5
    image_size: int = 224
    normalize_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    clearml_project_name: str = "Tazotron"
    clearml_task_name: str = "radiologynet_binary_necrosis_autopose"
    clearml_artifact_name: str = "best_model_state_dict"
    datasets_root_path: Path = Path("../.data")

    @property
    def data_root_path(self) -> Path:
        return self.datasets_root_path

    @property
    def datasets_catalog_path(self) -> Path:
        return self.data_root_path / "datasets"

    @property
    def models_root_path(self) -> Path:
        return self.data_root_path / "models"

    @property
    def weights_root_path(self) -> Path:
        return self.data_root_path / "weights"

@lru_cache(maxsize=1)
def get_settings() -> InferenceSettings:
    """Return cached service settings."""
    return InferenceSettings()
