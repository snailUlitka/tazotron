"""Utility helpers for working with ClearML tasks and datasets.

The helpers here were extracted from the training notebook so they can be reused
from regular Python modules or scripts.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from clearml import Dataset, Task
from clearml.logger import Logger

# Flag controlling whether a task should be kept in ClearML once finished.
CLEARML_SAVE_TASK: bool = True

# Defaults that reflect the original notebook configuration. Override as needed.
DEFAULT_PROJECT_NAME = "Tazotron"


def init_task(
    config: dict[str, Any],
    task_name: str,
    project_name: str = DEFAULT_PROJECT_NAME,
    tags: Sequence[str] | None = None,
) -> tuple[Task, Logger, dict[str, Any]]:
    """Create a ClearML task, attach the supplied configuration, and return its logger.

    Args:
        config: Arbitrary configuration dictionary that will be tracked by ClearML.
        project_name: Project to attach the task to.
        task_name: Name to display for the task in ClearML.
        tags: Optional iterable of tags. When omitted, the helper attempts to add the
            ``model_version`` from ``config`` (if present).

    Returns:
        A tuple of ``(task, logger, connected_config)`` where ``connected_config`` is
        the ClearML-tracked configuration returned by ``Task.connect_configuration``.
    """
    resolved_tags = list(tags) if tags is not None else []

    task = Task.init(project_name=project_name, task_name=task_name, tags=resolved_tags or None)
    logger = task.get_logger()
    connected_config = task.connect_configuration(config)
    return task, logger, connected_config


def get_dataset_path(dataset_project: str, dataset_name: str) -> Path:
    """Download (if necessary) and return the local path for a ClearML dataset."""
    dataset = Dataset.get(dataset_project=dataset_project, dataset_name=dataset_name)
    return Path(dataset.get_local_copy())


def upload_artifact(task: Task, alias: str, path: Path) -> None:
    """Upload a file or directory as a ClearML artifact."""
    task.upload_artifact(name=alias, artifact_object=str(path))


def finalize_task(task: Task, *, save_task: bool = CLEARML_SAVE_TASK) -> None:
    """Finish a ClearML task, mirroring the logic from the training notebook."""
    if save_task:
        task.mark_completed()
    else:
        task.close()
        task.set_archived(True)


def report_training_batch(
    logger: Logger,
    dataset_name: str,
    loss_value: float,
    iteration: int,
    ema_loss_value: float | None = None,
) -> None:
    """Log batch-level loss scalars for a dataset."""
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


def report_epoch_loss(logger: Logger, dataset_name: str, loss_value: float, epoch: int) -> None:
    """Log epoch-level loss scalars for a dataset."""
    logger.report_scalar(
        f"{dataset_name} Loss (Epochs)",
        "train_epoch",
        float(loss_value),
        iteration=epoch,
    )
