"""Tests for notebook ClearML helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from tazotron.integrations import clearml


class FakeLogger:
    """Simple logger spy for scalar calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.calls.append(
            {
                "title": title,
                "series": series,
                "value": value,
                "iteration": iteration,
            }
        )


class FakeTask:
    """Task spy with minimal ClearML API surface used by helpers."""

    init_calls: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def init(
        cls,
        project_name: str,
        task_name: str,
        tags: list[str] | None = None,
        *,
        auto_connect_frameworks: dict[str, bool] | bool | None = None,
    ) -> FakeTask:
        cls.init_calls.append(
            {
                "project_name": project_name,
                "task_name": task_name,
                "tags": tags,
                "auto_connect_frameworks": auto_connect_frameworks,
            }
        )
        return cls()

    def __init__(self) -> None:
        self.logger = FakeLogger()
        self.upload_calls: list[dict[str, Any]] = []
        self.config_calls: list[dict[str, Any]] = []
        self.mark_completed_calls = 0
        self.close_calls = 0
        self.set_archived_calls: list[bool] = []

    def get_logger(self) -> FakeLogger:
        return self.logger

    def connect_configuration(self, config: dict[str, Any]) -> dict[str, Any]:
        self.config_calls.append(config)
        return {"connected": config}

    def upload_artifact(
        self,
        name: str,
        artifact_object: str | Path,
        *,
        wait_on_upload: bool = False,
    ) -> None:
        self.upload_calls.append(
            {
                "name": name,
                "artifact_object": artifact_object,
                "wait_on_upload": wait_on_upload,
            }
        )

    def mark_completed(self) -> None:
        self.mark_completed_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def set_archived(self, value: bool) -> None:  # noqa: FBT001
        self.set_archived_calls.append(value)


class FakeDataset:
    """Dataset spy for get_dataset_path."""

    get_calls: ClassVar[list[dict[str, str]]] = []

    @classmethod
    def get(cls, dataset_project: str, dataset_name: str) -> FakeDataset:
        cls.get_calls.append({"dataset_project": dataset_project, "dataset_name": dataset_name})
        return cls()

    def get_local_copy(self) -> str:
        return "/tmp/fake-dataset"


def test_start_experiment_disabled_is_noop() -> None:
    run = clearml.start_experiment(
        config={"epochs": 2},
        task_name="disabled-test",
        enabled=False,
    )

    assert run.enabled is False
    assert run.task is None
    assert run.logger is None
    assert run.connected_config == {"epochs": 2}


def test_start_experiment_enabled_uses_clearml(monkeypatch: Any) -> None:
    FakeTask.init_calls = []
    monkeypatch.setattr(clearml, "_load_clearml", lambda: (FakeDataset, FakeTask))

    run = clearml.start_experiment(
        config={"lr": 1e-3},
        task_name="train-resnet18",
        project_name="Tazotron",
        tags=["resnet18"],
        enabled=True,
    )

    assert run.enabled is True
    assert isinstance(run.task, FakeTask)
    assert isinstance(run.logger, FakeLogger)
    assert run.connected_config == {"connected": {"lr": 1e-3}}
    assert FakeTask.init_calls == [
        {
            "project_name": "Tazotron",
            "task_name": "train-resnet18",
            "tags": ["resnet18"],
            "auto_connect_frameworks": {"pytorch": False},
        }
    ]
    assert run.task.config_calls == [{"lr": 1e-3}]


def test_logging_artifact_and_finalize_enabled(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(clearml, "_load_clearml", lambda: (FakeDataset, FakeTask))
    run = clearml.start_experiment(config={"seed": 42}, task_name="run", enabled=True)

    clearml.log_metrics(
        run=run,
        metrics={"loss": 0.123, "accuracy": 0.987},
        split="train",
        iteration=3,
        fold=2,
    )
    clearml.log_scalar(
        run=run,
        title="val_f1",
        series="val_fold_2",
        value=0.654,
        iteration=3,
    )

    checkpoint = tmp_path / "best_fold.pt"
    checkpoint.write_text("checkpoint")
    clearml.upload_model_artifact(run, alias="best_model", path=checkpoint)
    clearml.finish_experiment(run, save_task=False)

    assert run.logger is not None
    expected_logger_call_count = 3
    assert len(run.logger.calls) == expected_logger_call_count
    assert run.logger.calls[0] == {
        "title": "loss",
        "series": "train_fold_2",
        "value": 0.123,
        "iteration": 3,
    }
    assert run.logger.calls[1] == {
        "title": "accuracy",
        "series": "train_fold_2",
        "value": 0.987,
        "iteration": 3,
    }
    assert run.logger.calls[2] == {
        "title": "val_f1",
        "series": "val_fold_2",
        "value": 0.654,
        "iteration": 3,
    }

    assert run.task is not None
    assert run.task.upload_calls == [
        {"name": "best_model", "artifact_object": checkpoint.resolve(), "wait_on_upload": True}
    ]
    assert run.task.close_calls == 1
    assert run.task.set_archived_calls == [True]
    assert run.task.mark_completed_calls == 0


def test_upload_state_dict_artifact_serializes_and_uploads(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(clearml, "_load_clearml", lambda: (FakeDataset, FakeTask))
    run = clearml.start_experiment(config={"seed": 42}, task_name="run", enabled=True)

    class FakeModel:
        def state_dict(self) -> dict[str, int]:
            return {"weight": 1}

    artifact_path = clearml.upload_state_dict_artifact(
        run=run,
        alias="best_model_state_dict",
        model=FakeModel(),
        path=tmp_path / "best_fold_state_dict.pt",
    )

    assert artifact_path.is_file()
    assert run.task is not None
    assert run.task.upload_calls == [
        {
            "name": "best_model_state_dict",
            "artifact_object": artifact_path.resolve(),
            "wait_on_upload": True,
        }
    ]


def test_wrappers_respect_disabled_mode(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(clearml, "CLEARML_ENABLED", False)
    task, logger, config = clearml.init_task(config={"x": 1}, task_name="legacy")

    assert task is None
    assert logger is None
    assert config == {"x": 1}

    # All wrappers should be no-op when disabled, including with explicit objects.
    fake_task = FakeTask()
    fake_logger = FakeLogger()
    artifact = tmp_path / "legacy.pt"
    artifact.write_text("legacy")

    clearml.upload_artifact(fake_task, alias="legacy_model", path=artifact)
    clearml.finalize_task(fake_task)
    clearml.report_training_batch(fake_logger, dataset_name="xray", loss_value=0.1, iteration=1)
    clearml.report_epoch_loss(fake_logger, dataset_name="xray", loss_value=0.1, epoch=1)

    assert fake_task.upload_calls == []
    assert fake_task.mark_completed_calls == 0
    assert fake_task.close_calls == 0
    assert fake_task.set_archived_calls == []
    assert fake_logger.calls == []


def test_get_dataset_path_and_best_fold(monkeypatch: Any) -> None:
    FakeDataset.get_calls = []
    monkeypatch.setattr(clearml, "_load_clearml", lambda: (FakeDataset, FakeTask))

    dataset_path = clearml.get_dataset_path(dataset_project="xray", dataset_name="v1")
    assert dataset_path == Path("/tmp/fake-dataset")
    assert FakeDataset.get_calls == [{"dataset_project": "xray", "dataset_name": "v1"}]

    fold_index, fold_payload = clearml.select_best_fold(
        [
            {"fold": 1, "val": {"f1": 0.71}, "test": {"f1": 0.68}},
            {"fold": 2, "val": {"f1": 0.79}, "test": {"f1": 0.66}},
            {"fold": 3, "val": {"f1": 0.74}, "test": {"f1": 0.69}},
        ]
    )
    expected_best_fold_index = 2
    expected_best_fold_val_f1 = 0.79
    assert fold_index == expected_best_fold_index
    assert fold_payload["val"]["f1"] == expected_best_fold_val_f1
