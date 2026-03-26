"""Utility helpers for working with ClearML from training code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clearml import Task
    from clearml.logger import Logger

_Dataset: Any | None = None
_Task: Any | None = None

# Master switch for all ClearML side effects from this module.
CLEARML_ENABLED: bool = True

# Flag controlling whether a task should be kept in ClearML once finished.
CLEARML_SAVE_TASK: bool = True

# Defaults that reflect the original notebook configuration. Override as needed.
DEFAULT_PROJECT_NAME = "Tazotron"

# Disable ClearML PyTorch auto-hooking so dataset `.pt` files loaded with `torch.load`
# are not treated as framework artifacts.
DEFAULT_AUTO_CONNECT_FRAMEWORKS: dict[str, bool] = {"pytorch": False}


@dataclass(slots=True)
class ClearMLRun:
    """Container for a ClearML run. In disabled mode all fields stay empty."""

    enabled: bool
    task: Task | None = None
    logger: Logger | None = None
    connected_config: dict[str, Any] | None = None


def _resolve_enabled(enabled: bool | None = None) -> bool:  # noqa: FBT001
    """Resolve runtime flag against the module-level master switch."""
    return CLEARML_ENABLED if enabled is None else enabled


def _load_clearml() -> tuple[Any, Any]:
    """Resolve cached ClearML classes or raise a clear error if package is unavailable."""
    global _Dataset, _Task

    if _Dataset is None or _Task is None:
        try:
            from clearml import Dataset as dataset_cls
            from clearml import Task as task_cls
        except ImportError as error:
            msg = "clearml package is required when CLEARML_ENABLED=True"
            raise ImportError(msg) from error
        _Dataset = dataset_cls
        _Task = task_cls
    return _Dataset, _Task


def start_experiment(  # noqa: PLR0913
    config: dict[str, Any],
    task_name: str,
    project_name: str = DEFAULT_PROJECT_NAME,
    tags: list[str] | tuple[str, ...] | None = None,
    *,
    auto_connect_frameworks: dict[str, bool] | bool | None = None,
    enabled: bool | None = None,
) -> ClearMLRun:
    """Create and initialize a ClearML experiment if logging is enabled."""
    is_enabled = _resolve_enabled(enabled)
    if not is_enabled:
        return ClearMLRun(enabled=False, connected_config=dict(config))

    _, task_cls = _load_clearml()
    resolved_tags = list(tags) if tags is not None else []
    resolved_auto_connect_frameworks = (
        dict(DEFAULT_AUTO_CONNECT_FRAMEWORKS) if auto_connect_frameworks is None else auto_connect_frameworks
    )
    task = task_cls.init(
        project_name=project_name,
        task_name=task_name,
        tags=resolved_tags or None,
        auto_connect_frameworks=resolved_auto_connect_frameworks,
    )
    logger = task.get_logger()
    connected_config = task.connect_configuration(dict(config))
    return ClearMLRun(enabled=True, task=task, logger=logger, connected_config=connected_config)


def log_scalar(  # noqa: PLR0913
    run: ClearMLRun,
    title: str,
    series: str,
    value: float,
    *,
    iteration: int,
    enabled: bool | None = None,
) -> None:
    """Log a scalar metric in enabled mode."""
    if not _resolve_enabled(enabled) or not run.enabled or run.logger is None:
        return
    run.logger.report_scalar(title=title, series=series, value=float(value), iteration=iteration)


def log_metrics(  # noqa: PLR0913
    run: ClearMLRun,
    metrics: dict[str, float],
    *,
    split: str,
    iteration: int,
    fold: int | None = None,
    enabled: bool | None = None,
) -> None:
    """Log a dictionary of metrics under one split and optional fold."""
    if not _resolve_enabled(enabled) or not run.enabled or run.logger is None:
        return

    series = split if fold is None else f"{split}_fold_{fold}"
    for metric_name, metric_value in metrics.items():
        run.logger.report_scalar(
            title=metric_name,
            series=series,
            value=float(metric_value),
            iteration=iteration,
        )


def upload_model_artifact(
    run: ClearMLRun,
    alias: str,
    path: Path,
    *,
    wait_on_upload: bool = True,
    enabled: bool | None = None,
) -> None:
    """Upload a model checkpoint artifact in enabled mode."""
    if not _resolve_enabled(enabled) or not run.enabled or run.task is None:
        return
    path = path.expanduser().resolve()
    if not path.is_file():
        msg = f"Artifact path does not exist or is not a file: {path}"
        raise FileNotFoundError(msg)
    run.task.upload_artifact(
        name=alias,
        artifact_object=path,
        wait_on_upload=wait_on_upload,
    )


def upload_state_dict_artifact(  # noqa: PLR0913
    run: ClearMLRun,
    alias: str,
    model: Any,
    path: Path,
    *,
    wait_on_upload: bool = True,
    enabled: bool | None = None,
) -> Path:
    """Serialize a model state_dict to disk and upload it as a ClearML artifact."""
    resolved_path = path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    if not isinstance(state_dict, dict):
        msg = "model.state_dict() must return a dictionary-like object"
        raise TypeError(msg)
    import torch  # noqa: PLC0415

    torch.save(state_dict, resolved_path)
    upload_model_artifact(
        run=run,
        alias=alias,
        path=resolved_path,
        wait_on_upload=wait_on_upload,
        enabled=enabled,
    )
    return resolved_path


def finish_experiment(
    run: ClearMLRun,
    *,
    save_task: bool = CLEARML_SAVE_TASK,
    enabled: bool | None = None,
) -> None:
    """Finalize a ClearML experiment in enabled mode."""
    if not _resolve_enabled(enabled) or not run.enabled or run.task is None:
        return
    if save_task:
        run.task.mark_completed()
    else:
        run.task.close()
        run.task.set_archived(True)


def select_best_fold(
    fold_results: list[dict[str, Any]],
    *,
    split: str = "val",
    metric: str = "f1",
) -> tuple[int, dict[str, Any]]:
    """Return 1-based fold index and payload for the best fold by split/metric."""
    if not fold_results:
        msg = "fold_results must not be empty"
        raise ValueError(msg)

    best_position = max(
        range(1, len(fold_results) + 1),
        key=lambda position: float(fold_results[position - 1][split][metric]),
    )
    return best_position, fold_results[best_position - 1]


def init_task(  # noqa: PLR0913
    config: dict[str, Any],
    task_name: str,
    project_name: str = DEFAULT_PROJECT_NAME,
    tags: list[str] | tuple[str, ...] | None = None,
    *,
    auto_connect_frameworks: dict[str, bool] | bool | None = None,
    enabled: bool | None = None,
) -> tuple[Task | None, Logger | None, dict[str, Any]]:
    """Backward-compatible wrapper for creating a task and logger."""
    run = start_experiment(
        config=config,
        task_name=task_name,
        project_name=project_name,
        tags=tags,
        auto_connect_frameworks=auto_connect_frameworks,
        enabled=enabled,
    )
    return run.task, run.logger, run.connected_config or dict(config)


def get_dataset_path(dataset_project: str, dataset_name: str) -> Path:
    """Download (if necessary) and return the local path for a ClearML dataset."""
    dataset_cls, _ = _load_clearml()
    dataset = dataset_cls.get(dataset_project=dataset_project, dataset_name=dataset_name)
    return Path(dataset.get_local_copy())


def get_task(project_name: str, task_name: str) -> Any:
    """Resolve a ClearML task by project and task name."""
    _, task_cls = _load_clearml()
    task = task_cls.get_task(project_name=project_name, task_name=task_name)
    if task is None:
        msg = f"ClearML task '{task_name}' was not found in project '{project_name}'"
        raise ValueError(msg)
    return task


def get_task_artifact_path(task: Task | Any, alias: str) -> Path:
    """Return a local path for a task artifact."""
    artifacts = getattr(task, "artifacts", None)
    if not isinstance(artifacts, dict):
        msg = "ClearML task does not expose an artifacts dictionary"
        raise ValueError(msg)
    artifact = artifacts.get(alias)
    if artifact is None:
        msg = f"ClearML artifact '{alias}' was not found for task"
        raise KeyError(msg)
    local_copy = artifact.get_local_copy()
    if local_copy is None:
        msg = f"ClearML artifact '{alias}' did not provide a local copy"
        raise RuntimeError(msg)
    return Path(local_copy)


def upload_artifact(task: Task | None, alias: str, path: Path, *, enabled: bool | None = None) -> None:
    """Backward-compatible helper for uploading an artifact by task object."""
    if not _resolve_enabled(enabled) or task is None:
        return
    task.upload_artifact(name=alias, artifact_object=str(path))


def finalize_task(
    task: Task | None,
    *,
    save_task: bool = CLEARML_SAVE_TASK,
    enabled: bool | None = None,
) -> None:
    """Backward-compatible helper for finalizing an existing task."""
    if not _resolve_enabled(enabled) or task is None:
        return
    if save_task:
        task.mark_completed()
    else:
        task.close()
        task.set_archived(True)


def report_training_batch(  # noqa: PLR0913
    logger: Logger | None,
    dataset_name: str,
    loss_value: float,
    iteration: int,
    ema_loss_value: float | None = None,
    *,
    enabled: bool | None = None,
) -> None:
    """Log batch-level loss scalars for a dataset."""
    if not _resolve_enabled(enabled) or logger is None:
        return
    logger.report_scalar(
        f"{dataset_name} Loss",
        "train_batch",
        float(loss_value),
        iteration=iteration,
    )
    if ema_loss_value is not None:
        logger.report_scalar(
            f"{dataset_name} Loss",
            "train_batch_ema",
            float(ema_loss_value),
            iteration=iteration,
        )


def report_epoch_loss(
    logger: Logger | None,
    dataset_name: str,
    loss_value: float,
    epoch: int,
    *,
    enabled: bool | None = None,
) -> None:
    """Log epoch-level loss scalars for a dataset."""
    if not _resolve_enabled(enabled) or logger is None:
        return
    logger.report_scalar(
        f"{dataset_name} Loss (Epochs)",
        "train_epoch",
        float(loss_value),
        iteration=epoch,
    )
