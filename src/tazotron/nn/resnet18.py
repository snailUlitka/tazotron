"""Lightweight ResNet18-style classifier for XR TIFF inputs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Subset

if TYPE_CHECKING:
    from collections.abc import Callable


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
    label_key: str = "label"
    seed: int = 42
    progress_fn: Callable[[str], None] | None = None


class ResNet18(nn.Module):
    """Compact ResNet18-like CNN tailored for grayscale XR inputs."""

    def __init__(self, num_classes: int, *, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass."""
        x = self.features(x)
        return self.classifier(x)

    @staticmethod
    def _prepare_inputs(x: torch.Tensor, image_size: int) -> torch.Tensor:
        """Convert XR tensor to 3-channel float tensor resized to the requested square size."""
        if x.dim() == SQUEEZE_RANK:
            x = x.squeeze(1)
        if x.ndim != BATCH_RANK:
            msg = f"Expected input with 4 dims (B, C, H, W) after squeeze, got shape {tuple(x.shape)}"
            raise ValueError(msg)
        if x.shape[1] == CHANNELS_GRAY:
            x = x.repeat(1, CHANNELS_RGB, 1, 1)
        elif x.shape[1] != CHANNELS_RGB:
            msg = f"Expected 1 or 3 channels, got {x.shape[1]}"
            raise ValueError(msg)
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

        inputs = ResNet18._prepare_inputs(inputs.float(), image_size=image_size).to(device)
        if targets.dim() > 1:
            targets = targets.squeeze()
        targets = targets.to(device=device, dtype=torch.long)
        return inputs, targets

    @staticmethod
    def _build_loader(
        dataset: Dataset[Any],
        *,
        base_loader: DataLoader[Any],
        shuffle: bool,
        num_workers: int | None,
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=base_loader.batch_size,
            shuffle=shuffle,
            num_workers=num_workers if num_workers is not None else base_loader.num_workers,
            pin_memory=base_loader.pin_memory,
            drop_last=base_loader.drop_last,
            collate_fn=base_loader.collate_fn,
        )

    @staticmethod
    def train_(
        model: ResNet18,
        dataloader: DataLoader[Any],
        config: TrainingConfig,
    ) -> dict[str, Any]:
        """Train model with k-fold cross-validation and early stopping."""
        device = torch.device(config.device)

        ResNet18._ensure_model_compatible(model, config)

        base_state = copy.deepcopy(model.state_dict())
        dataset = dataloader.dataset
        dataset_size = len(dataset)
        if dataset_size == 0:
            msg = "Dataset is empty; cannot run cross-validation."
            raise ValueError(msg)
        folds = min(config.folds, dataset_size)
        generator = torch.Generator().manual_seed(config.seed)
        shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()
        fold_sizes = [dataset_size // folds for _ in range(folds)]
        for i in range(dataset_size % folds):
            fold_sizes[i] += 1
        fold_results: list[dict[str, Any]] = []
        idx_cursor = 0

        for fold_idx, fold_size in enumerate(fold_sizes):
            val_indices = shuffled_indices[idx_cursor : idx_cursor + fold_size]
            train_indices = shuffled_indices[:idx_cursor] + shuffled_indices[idx_cursor + fold_size :]
            idx_cursor += fold_size

            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            if len(train_subset) == 0:
                msg = "Not enough samples to create a training split; reduce fold count or add data."
                raise ValueError(msg)
            train_loader = ResNet18._build_loader(
                train_subset,
                base_loader=dataloader,
                shuffle=True,
                num_workers=config.num_workers,
            )
            val_loader = ResNet18._build_loader(
                val_subset,
                base_loader=dataloader,
                shuffle=False,
                num_workers=config.num_workers,
            )

            fold_results.append(
                ResNet18._run_fold(
                    model=model,
                    base_state=base_state,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    config=config,
                    fold_idx=fold_idx,
                    fold_count=folds,
                    train_size=len(train_subset),
                    val_size=len(val_subset),
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
        model: ResNet18,
        dataloader: DataLoader[Any],
        config: TrainingConfig,
    ) -> dict[str, Any]:
        """Evaluate trained model on a test loader."""
        device = torch.device(config.device)
        ResNet18._ensure_model_compatible(model, config)
        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = ResNet18._extract_batch(
                    batch,
                    label_key=config.label_key,
                    image_size=config.image_size,
                    device=device,
                )
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()

        dataset_size = len(dataloader.dataset)
        mean_loss = total_loss / max(dataset_size, 1)
        accuracy = correct / max(total, 1)
        if config.progress_fn:
            config.progress_fn(f"Test: loss={mean_loss:.4f}, acc={accuracy:.4f}")
        return {
            "loss": mean_loss,
            "accuracy": accuracy,
            "samples": dataset_size,
        }

    @staticmethod
    def _run_fold(  # noqa: PLR0913
        *,
        model: ResNet18,
        base_state: dict[str, torch.Tensor],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        device: torch.device,
        config: TrainingConfig,
        fold_idx: int,
        fold_count: int,
        train_size: int,
        val_size: int,
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
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = ResNet18._extract_batch(
                    batch,
                    label_key=config.label_key,
                    image_size=config.image_size,
                    device=device,
                )
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_train_loss = running_loss / train_size
            train_history.append(epoch_train_loss)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = ResNet18._extract_batch(
                        batch,
                        label_key=config.label_key,
                        image_size=config.image_size,
                        device=device,
                    )
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.numel()

            epoch_val_loss = val_loss / val_size if val_size else float("inf")
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
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
    def _ensure_model_compatible(model: ResNet18, config: TrainingConfig) -> None:
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
