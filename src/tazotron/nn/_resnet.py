"""Shared configurable ResNet-style models and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


BATCH_RANK = 4
CHANNELS_GRAY = 1
CHANNELS_RGB = 3
SQUEEZE_RANK = 5


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters and training knobs for k-fold training."""

    num_classes: int
    folds: int = 5
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 0.0
    image_size: int = 128
    dropout: float = 0.3
    device: str | torch.device = "cpu"
    num_workers: int | None = None
    label_key: str = "label_combined_femoral_head"
    seed: int = 42
    progress_fn: Callable[[str], None] | None = None


def build_feature_extractor(
    stage_channels: Sequence[Sequence[int]],
    *,
    pool_after_stages: Sequence[int],
) -> tuple[nn.Sequential, int]:
    """Build a sequential convolutional feature extractor from stage specs."""
    layers: list[nn.Module] = []
    in_channels = CHANNELS_RGB
    pooling_stages = set(pool_after_stages)
    last_channels = in_channels

    for stage_index, stage in enumerate(stage_channels):
        for out_channels in stage:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
            last_channels = out_channels
        if stage_index in pooling_stages:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    return nn.Sequential(*layers), last_channels


def build_classifier_head(
    in_features: int,
    *,
    hidden_features: int,
    num_classes: int,
    dropout: float,
) -> nn.Sequential:
    """Build the classifier stack shared across custom models."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, hidden_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_features, num_classes),
    )


class ResNetClassifierBase(nn.Module):
    """Shared base for custom ResNet-style classifiers."""

    stage_channels: Sequence[Sequence[int]]
    pool_after_stages: Sequence[int]
    classifier_hidden_features: int

    def __init__(self, num_classes: int, *, dropout: float = 0.3) -> None:
        super().__init__()
        self.features, feature_dim = build_feature_extractor(
            self.stage_channels,
            pool_after_stages=self.pool_after_stages,
        )
        self.classifier = build_classifier_head(
            feature_dim,
            hidden_features=self.classifier_hidden_features,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        x = self._coerce_input_channels(x)
        return self.classifier(self.features(x))

    @staticmethod
    def _coerce_input_channels(x: torch.Tensor) -> torch.Tensor:
        """Accept grayscale or RGB tensors in direct forward passes."""
        if x.dim() == SQUEEZE_RANK:
            x = x.squeeze(1)
        if x.ndim != BATCH_RANK:
            msg = f"Expected input with 4 dims (B, C, H, W) after squeeze, got shape {tuple(x.shape)}"
            raise ValueError(msg)
        if x.shape[1] == CHANNELS_GRAY:
            return x.repeat(1, CHANNELS_RGB, 1, 1)
        if x.shape[1] != CHANNELS_RGB:
            msg = f"Expected 1 or 3 channels, got {x.shape[1]}"
            raise ValueError(msg)
        return x

    @staticmethod
    def _prepare_inputs(x: torch.Tensor, image_size: int) -> torch.Tensor:
        """Convert XR tensor to 3-channel float tensor resized to the requested square size."""
        x = ResNetClassifierBase._coerce_input_channels(x)
        if x.shape[2] != image_size or x.shape[3] != image_size:
            x = functional.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _extract_batch(
        batch: Any,
        *,
        label_key: str,
        image_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn a mixed batch (dict/tuple) into model-ready tensors."""
        if isinstance(batch, dict):
            inputs = batch.get("drr")
            targets = batch.get(label_key)
        elif isinstance(batch, (list, tuple)) and batch:
            candidate = batch[0]
            if isinstance(candidate, dict):
                inputs_list = [sample["drr"] for sample in batch]
                target_list = [sample[label_key] for sample in batch]
                inputs = torch.stack(inputs_list)
                targets = torch.stack([t if isinstance(t, torch.Tensor) else torch.as_tensor(t) for t in target_list])
            else:
                inputs, targets = batch  # type: ignore[misc]
        else:
            msg = f"Unsupported batch type: {type(batch)}"
            raise TypeError(msg)

        if inputs is None or targets is None:
            msg = f"Batch must provide 'drr' and '{label_key}' tensors."
            raise KeyError(msg)

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs)
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        inputs = ResNetClassifierBase._prepare_inputs(inputs.float(), image_size=image_size).to(device)
        targets = targets.reshape(-1).to(device=device, dtype=torch.long)
        return inputs, targets

    @staticmethod
    def _build_loader(
        dataset: Dataset[Any],
        *,
        base_loader: DataLoader[Any],
        shuffle: bool,
        num_workers: int | None,
    ) -> DataLoader[Any]:
        """Clone a base dataloader with a different dataset split."""
        sampler = None
        shuffle_flag = shuffle
        base_sampler = base_loader.sampler
        if isinstance(base_sampler, torch.utils.data.WeightedRandomSampler):
            if isinstance(dataset, Subset) and isinstance(base_loader.dataset, Dataset):
                weights = base_sampler.weights[dataset.indices]
            else:
                weights = base_sampler.weights
            num_samples = base_sampler.num_samples
            if not base_sampler.replacement and num_samples > len(weights):
                num_samples = len(weights)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=num_samples,
                replacement=base_sampler.replacement,
                generator=base_sampler.generator,
            )
            shuffle_flag = False
        elif isinstance(base_sampler, DistributedSampler):
            sampler = DistributedSampler(
                dataset,
                num_replicas=base_sampler.num_replicas,
                rank=base_sampler.rank,
                shuffle=shuffle,
                seed=base_sampler.seed,
                drop_last=base_sampler.drop_last,
            )
            shuffle_flag = False
        elif not isinstance(
            base_sampler,
            (torch.utils.data.RandomSampler, torch.utils.data.SequentialSampler),
        ):
            msg = (
                "Unsupported custom sampler in base_loader; provide a loader without a custom sampler or extend "
                "_build_loader to handle it."
            )
            raise ValueError(msg)
        return DataLoader(
            dataset,
            batch_size=base_loader.batch_size,
            shuffle=shuffle_flag,
            sampler=sampler,
            num_workers=num_workers if num_workers is not None else base_loader.num_workers,
            pin_memory=base_loader.pin_memory,
            drop_last=base_loader.drop_last,
            collate_fn=base_loader.collate_fn,
            timeout=base_loader.timeout,
            worker_init_fn=base_loader.worker_init_fn,
            persistent_workers=base_loader.persistent_workers,
            prefetch_factor=base_loader.prefetch_factor,
            generator=base_loader.generator,
        )

    @staticmethod
    def train_(
        model: ResNetClassifierBase,
        dataloader: DataLoader[Any],
        config: TrainingConfig,
    ) -> dict[str, Any]:
        """Train model with k-fold cross-validation and early stopping."""
        device = torch.device(config.device)
        ResNetClassifierBase._ensure_model_compatible(model, config)

        base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        dataset = dataloader.dataset
        dataset_size = len(dataset)
        if dataset_size == 0:
            msg = "Dataset is empty; cannot run cross-validation."
            raise ValueError(msg)

        folds = min(config.folds, dataset_size)
        generator = torch.Generator().manual_seed(config.seed)
        shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()
        fold_sizes = [dataset_size // folds for _ in range(folds)]
        for index in range(dataset_size % folds):
            fold_sizes[index] += 1

        fold_results: list[dict[str, Any]] = []
        index_cursor = 0
        for fold_idx, fold_size in enumerate(fold_sizes):
            val_indices = shuffled_indices[index_cursor : index_cursor + fold_size]
            train_indices = shuffled_indices[:index_cursor] + shuffled_indices[index_cursor + fold_size :]
            index_cursor += fold_size

            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            if len(train_subset) == 0:
                msg = "Not enough samples to create a training split; reduce fold count or add data."
                raise ValueError(msg)

            train_loader = ResNetClassifierBase._build_loader(
                train_subset,
                base_loader=dataloader,
                shuffle=True,
                num_workers=config.num_workers,
            )
            val_loader = ResNetClassifierBase._build_loader(
                val_subset,
                base_loader=dataloader,
                shuffle=False,
                num_workers=config.num_workers,
            )
            fold_results.append(
                ResNetClassifierBase._run_fold(
                    model=model,
                    base_state=base_state,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    config=config,
                    fold_idx=fold_idx,
                    fold_count=folds,
                )
            )

        best_acc = [max(result["val_accuracy"]) if result["val_accuracy"] else 0.0 for result in fold_results]
        best_loss = [result["best_val_loss"] for result in fold_results]
        return {
            "folds": fold_results,
            "fold_count": folds,
            "config": config,
            "mean_best_val_accuracy": sum(best_acc) / len(best_acc) if best_acc else 0.0,
            "mean_best_val_loss": sum(best_loss) / len(best_loss) if best_loss else float("inf"),
        }

    @staticmethod
    def test_(
        model: ResNetClassifierBase,
        dataloader: DataLoader[Any],
        config: TrainingConfig,
    ) -> dict[str, Any]:
        """Evaluate trained model on a test loader."""
        device = torch.device(config.device)
        ResNetClassifierBase._ensure_model_compatible(model, config)
        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = ResNetClassifierBase._extract_batch(
                    batch,
                    label_key=config.label_key,
                    image_size=config.image_size,
                    device=device,
                )
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()

        mean_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        if config.progress_fn:
            config.progress_fn(f"Test: loss={mean_loss:.4f}, acc={accuracy:.4f}")
        return {
            "loss": mean_loss,
            "accuracy": accuracy,
            "samples": len(dataloader.dataset),
        }

    @staticmethod
    def _run_fold(  # noqa: PLR0913
        *,
        model: ResNetClassifierBase,
        base_state: dict[str, torch.Tensor],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: torch.device,
        config: TrainingConfig,
        fold_idx: int,
        fold_count: int,
    ) -> dict[str, Any]:
        """Train one fold and return metrics."""
        model.load_state_dict(base_state)
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_no_improve = 0
        train_history: list[float] = []
        val_history: list[float] = []
        val_accuracy: list[float] = []

        for epoch in range(config.num_epochs):
            model.train()
            running_loss = 0.0
            seen = 0
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = ResNetClassifierBase._extract_batch(
                    batch,
                    label_key=config.label_key,
                    image_size=config.image_size,
                    device=device,
                )
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                seen += batch_size

            epoch_train_loss = running_loss / max(seen, 1)
            train_history.append(epoch_train_loss)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = ResNetClassifierBase._extract_batch(
                        batch,
                        label_key=config.label_key,
                        image_size=config.image_size,
                        device=device,
                    )
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    batch_size = inputs.size(0)
                    val_loss += loss.item() * batch_size
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.numel()

            epoch_val_loss = val_loss / max(total, 1)
            epoch_val_acc = correct / max(total, 1)
            val_history.append(epoch_val_loss)
            val_accuracy.append(epoch_val_acc)

            if config.progress_fn:
                config.progress_fn(
                    f"Fold {fold_idx + 1}/{fold_count}, epoch {epoch + 1}: "
                    f"train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}, "
                    f"val_acc={epoch_val_acc:.4f}"
                )

            if epoch_val_loss < best_val_loss - config.min_delta:
                best_val_loss = epoch_val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        return {
            "fold": fold_idx,
            "train_loss": train_history,
            "val_loss": val_history,
            "val_accuracy": val_accuracy,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_history),
        }

    @staticmethod
    def _ensure_model_compatible(model: ResNetClassifierBase, config: TrainingConfig) -> None:
        """Validate head size and align dropout probability with config."""
        final_layer = None
        if isinstance(getattr(model, "classifier", None), nn.Sequential) and len(model.classifier) > 0:
            maybe_linear = list(model.classifier)[-1]
            if isinstance(maybe_linear, nn.Linear):
                final_layer = maybe_linear
        if final_layer and final_layer.out_features != config.num_classes:
            msg = (
                "TrainingConfig.num_classes does not match model output features: "
                f"{config.num_classes} vs {final_layer.out_features}"
            )
            raise ValueError(msg)

        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = config.dropout
