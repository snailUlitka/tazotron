"""Benchmark legacy crop rendering against autopose rendering and downstream training."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from tazotron.benchmarks.xray_pose import (
    build_case_level_split_rows,
    compare_render_reports,
    diff_to_rgb_image,
    summarize_render_rows,
    tensor_to_grayscale_image,
    write_csv_rows,
    write_json,
)
from tazotron.datasets.ct import CTDataset
from tazotron.datasets.transforms.xray import RenderDRR
from tazotron.nn.resnet18 import ResNet18
from tazotron.xray_generation import (
    AUTOPOSE_MODE,
    LEGACY_CROP_MODE,
    apply_framing,
    benchmark_render_mode,
    render_xray_dataset_from_ct,
    rows_to_dicts,
    squeeze_xray_tensor,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torchio as tio

DEFAULT_TEST_RATIO = 0.10
DEFAULT_FOLDS = 5
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 8
DEFAULT_IMAGE_SIZE = 224
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.3
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42


class LabeledPathDataset(Dataset[dict[str, object]]):
    """Case-level labeled XR dataset backed by `.pt` files."""

    def __init__(self, entries: Sequence[dict[str, object]], transform: transforms.Compose | None) -> None:
        self.entries = list(entries)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        tensor = _load_xray_tensor(Path(str(entry["path"])))
        if self.transform is not None:
            tensor = self.transform(tensor)
        return {
            "image": tensor,
            "label": torch.tensor(int(entry["label"]), dtype=torch.long),
            "case_id": str(entry["case_id"]),
            "path": str(entry["path"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".data/Totalsegmentator_dataset"),
        help="CT dataset root with per-case folders.",
    )
    parser.add_argument(
        "--before-dataset-root",
        type=Path,
        default=Path(".data/output_with_crop"),
        help="Legacy crop XR dataset root used for downstream-before.",
    )
    parser.add_argument(
        "--after-dataset-root",
        type=Path,
        default=Path(".data/output_with_autopose"),
        help="Autopose XR dataset root; will be regenerated in place.",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports/xray_pose_before_after"),
        help="Directory that will contain benchmark reports.",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        help="Report subdirectory name.",
    )
    parser.add_argument("--device", type=str, default=_default_device(), help="Torch device for rendering/training.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Input resize for the classifier.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Dataloader worker count.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for splitting/training.")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Hold-out test ratio.")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Number of train/val folds on trainval.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Classifier dropout probability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    report_dir = args.reports_root / args.run_label
    before_dir = report_dir / "before"
    after_dir = report_dir / "after"
    comparison_dir = report_dir / "comparison"
    samples_dir = report_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    before_render_metrics_path = before_dir / "render_metrics.csv"
    after_render_metrics_path = after_dir / "render_metrics.csv"
    before_render_summary_path = before_dir / "render_summary.json"
    after_render_summary_path = after_dir / "render_summary.json"
    render_compare_path = comparison_dir / "render_compare.csv"
    render_compare_summary_path = comparison_dir / "render_compare_summary.json"
    case_split_path = comparison_dir / "case_split.csv"

    if _render_reports_exist(
        before_render_metrics_path=before_render_metrics_path,
        after_render_metrics_path=after_render_metrics_path,
        before_render_summary_path=before_render_summary_path,
        after_render_summary_path=after_render_summary_path,
        render_compare_path=render_compare_path,
        render_compare_summary_path=render_compare_summary_path,
    ):
        before_summary = _read_json(before_render_summary_path)
        after_summary = _read_json(after_render_summary_path)
        comparison_rows = _read_csv_rows(render_compare_path)
        comparison_summary = _read_json(render_compare_summary_path)
    else:
        if not args.before_dataset_root.is_dir():
            render_xray_dataset_from_ct(
                args.data_root,
                args.before_dataset_root,
                framing_mode=LEGACY_CROP_MODE,
                device=args.device,
            )
        if not args.after_dataset_root.is_dir() or not _paired_case_ids(args.after_dataset_root):
            render_xray_dataset_from_ct(
                args.data_root,
                args.after_dataset_root,
                framing_mode=AUTOPOSE_MODE,
                device=args.device,
                overwrite_existing=True,
            )

        before_rows = benchmark_render_mode(args.data_root, framing_mode=LEGACY_CROP_MODE, device=args.device)
        after_rows = benchmark_render_mode(args.data_root, framing_mode=AUTOPOSE_MODE, device=args.device)
        before_summary = summarize_render_rows(before_rows)
        after_summary = summarize_render_rows(after_rows)
        comparison_rows, comparison_summary = compare_render_reports(before_rows, after_rows)

        write_csv_rows(before_render_metrics_path, rows_to_dicts(before_rows))
        write_csv_rows(after_render_metrics_path, rows_to_dicts(after_rows))
        write_json(before_render_summary_path, before_summary)
        write_json(after_render_summary_path, after_summary)
        write_csv_rows(render_compare_path, comparison_rows)
        write_json(render_compare_summary_path, comparison_summary)

    common_case_ids = sorted(_paired_case_ids(args.before_dataset_root) & _paired_case_ids(args.after_dataset_root))
    if not common_case_ids:
        msg = "No common paired case ids found between before/after XR datasets."
        raise ValueError(msg)
    if case_split_path.is_file():
        split_rows = _read_csv_rows(case_split_path)
    else:
        split_rows = build_case_level_split_rows(
            common_case_ids,
            test_ratio=args.test_ratio,
            folds=args.folds,
            seed=args.seed,
        )
        write_csv_rows(case_split_path, split_rows)

    before_downstream = _run_downstream_benchmark(
        args.before_dataset_root,
        split_rows=split_rows,
        report_dir=before_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )
    after_downstream = _run_downstream_benchmark(
        args.after_dataset_root,
        split_rows=split_rows,
        report_dir=after_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )

    _save_sample_triplets(
        data_root=args.data_root,
        comparison_rows=comparison_rows,
        samples_dir=samples_dir,
        device=args.device,
    )
    _write_summary_markdown(
        report_dir / "summary.md",
        before_render=before_summary,
        after_render=after_summary,
        comparison=comparison_summary,
        before_downstream=before_downstream,
        after_downstream=after_downstream,
        run_label=args.run_label,
    )

    print(f"Reports saved to: {report_dir}")


def _run_downstream_benchmark(  # noqa: PLR0913
    dataset_root: Path,
    *,
    split_rows: list[dict[str, object]],
    report_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    num_workers: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
) -> dict[str, object]:
    _seed_everything(seed)

    folds = sorted({int(row["fold"]) for row in split_rows if row["split"] == "trainval"})
    test_case_ids = sorted(str(row["case_id"]) for row in split_rows if row["split"] == "test")
    if not folds:
        msg = "Split manifest does not contain trainval folds."
        raise ValueError(msg)
    if not test_case_ids:
        msg = "Split manifest does not contain a hold-out test split."
        raise ValueError(msg)

    train_transform, eval_transform = _build_transforms(image_size)
    fold_results: list[dict[str, object]] = []

    for fold_idx in folds:
        train_case_ids = sorted(
            str(row["case_id"]) for row in split_rows if row["split"] == "trainval" and int(row["fold"]) != fold_idx
        )
        val_case_ids = sorted(
            str(row["case_id"]) for row in split_rows if row["split"] == "trainval" and int(row["fold"]) == fold_idx
        )

        train_entries = _build_entries(dataset_root, train_case_ids)
        val_entries = _build_entries(dataset_root, val_case_ids)
        test_entries = _build_entries(dataset_root, test_case_ids)
        if not train_entries or not val_entries or not test_entries:
            msg = (
                "One of the downstream splits is empty. "
                f"train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}"
            )
            raise ValueError(msg)

        train_loader, val_loader, test_loader = _build_loaders(
            train_entries,
            val_entries,
            test_entries,
            train_transform=train_transform,
            eval_transform=eval_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.startswith("cuda"),
            seed=seed + fold_idx,
        )

        model = ResNet18(num_classes=2, dropout=dropout).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        train_history = _train_one_fold(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
        )
        val_metrics = _evaluate(model, val_loader, device=device, desc=f"val-{fold_idx}")
        test_metrics = _evaluate(model, test_loader, device=device, desc=f"test-{fold_idx}")
        fold_results.append(
            {
                "fold": fold_idx,
                "train_cases": len(train_case_ids),
                "val_cases": len(val_case_ids),
                "test_cases": len(test_case_ids),
                **train_history,
                "val": val_metrics,
                "test": test_metrics,
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "fold_count": len(folds),
        "folds": fold_results,
        "aggregate": _aggregate_fold_results(fold_results),
    }
    write_json(report_dir / "downstream.json", summary)
    return summary


def _train_one_fold(  # noqa: PLR0913
    model: torch.nn.Module,
    train_loader: DataLoader[dict[str, object]],
    val_loader: DataLoader[dict[str, object]],
    criterion: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: str,
    epochs: int,
) -> dict[str, object]:
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device, is_train=True)
        val_loss = _run_epoch(model, val_loader, criterion, optimizer=None, device=device, is_train=False)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "epochs_trained": epochs,
        "best_val_loss": best_val_loss,
        "train_loss": train_losses,
        "val_loss": val_losses,
    }


def _run_epoch(  # noqa: PLR0913
    model: torch.nn.Module,
    loader: DataLoader[dict[str, object]],
    criterion: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    is_train: bool,
) -> float:
    model.train(mode=is_train)
    total_loss = 0.0
    total = 0
    for batch in loader:
        inputs = batch["image"].to(device)
        targets = batch["label"].to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if is_train:
                if optimizer is None:
                    msg = "optimizer is required when is_train=True."
                    raise ValueError(msg)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total += targets.size(0)
    return total_loss / max(1, total)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, object]],
    *,
    device: str,
    desc: str,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            tp += ((preds == 1) & (targets == 1)).sum().item()
            fp += ((preds == 1) & (targets == 0)).sum().item()
            fn += ((preds == 0) & (targets == 1)).sum().item()
    accuracy = correct / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-8, precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _aggregate_fold_results(fold_results: list[dict[str, object]]) -> dict[str, float]:
    if not fold_results:
        msg = "fold_results must not be empty."
        raise ValueError(msg)

    def values(metric: str, split: str) -> torch.Tensor:
        return torch.tensor([float(result[split][metric]) for result in fold_results], dtype=torch.float32)  # type: ignore[index]

    metrics = {}
    for split in ("val", "test"):
        for metric in ("accuracy", "precision", "recall", "f1"):
            tensor = values(metric, split)
            metrics[f"{split}_{metric}_mean"] = float(tensor.mean().item())
            metrics[f"{split}_{metric}_std"] = float(tensor.std(unbiased=False).item())
    return metrics


def _build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Lambda(_to_three_channels),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Lambda(_to_three_channels),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return train_transform, eval_transform


def _build_loaders(  # noqa: PLR0913
    train_entries: list[dict[str, object]],
    val_entries: list[dict[str, object]],
    test_entries: list[dict[str, object]],
    *,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], DataLoader[dict[str, object]]]:
    train_dataset = LabeledPathDataset(train_entries, train_transform)
    val_dataset = LabeledPathDataset(val_entries, eval_transform)
    test_dataset = LabeledPathDataset(test_entries, eval_transform)
    generator = torch.Generator().manual_seed(seed)
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


def _build_entries(dataset_root: Path, case_ids: Sequence[str]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for case_id in case_ids:
        healthy_path = dataset_root / "without_necro" / f"{case_id}.pt"
        necro_path = dataset_root / "with_necro" / f"{case_id}.pt"
        if not healthy_path.is_file() or not necro_path.is_file():
            continue
        entries.append({"case_id": case_id, "label": 0, "path": healthy_path})
        entries.append({"case_id": case_id, "label": 1, "path": necro_path})
    return entries


def _paired_case_ids(dataset_root: Path) -> set[str]:
    with_necro = {path.stem for path in (dataset_root / "with_necro").glob("*.pt")}
    without_necro = {path.stem for path in (dataset_root / "without_necro").glob("*.pt")}
    return with_necro & without_necro


def _render_reports_exist(  # noqa: PLR0913
    *,
    before_render_metrics_path: Path,
    after_render_metrics_path: Path,
    before_render_summary_path: Path,
    after_render_summary_path: Path,
    render_compare_path: Path,
    render_compare_summary_path: Path,
) -> bool:
    return all(
        path.is_file()
        for path in (
            before_render_metrics_path,
            after_render_metrics_path,
            before_render_summary_path,
            after_render_summary_path,
            render_compare_path,
            render_compare_summary_path,
        )
    )


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_xray_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        msg = f"Expected a torch.Tensor in {path}"
        raise TypeError(msg)
    image = squeeze_xray_tensor(tensor)
    return image.unsqueeze(0).to(torch.float32)


def _to_three_channels(image: torch.Tensor) -> torch.Tensor:
    return image.expand(3, -1, -1)


def _save_sample_triplets(
    *,
    data_root: Path,
    comparison_rows: list[dict[str, object]],
    samples_dir: Path,
    device: str,
) -> None:
    sample_case_ids = _select_sample_case_ids(comparison_rows)
    if not sample_case_ids:
        return

    dataset = CTDataset(data_root)
    index_by_case = {path.parent.name: index for index, path in enumerate(dataset.paths)}
    for case_id in tqdm(sample_case_ids, desc="Saving samples", leave=False):
        index = index_by_case.get(case_id)
        if index is None:
            continue
        sample_dir = samples_dir / case_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        subject = dataset[index]
        before_tensor = _render_clean_case(subject, framing_mode=LEGACY_CROP_MODE, device=device)
        after_tensor = _render_clean_case(subject, framing_mode=AUTOPOSE_MODE, device=device)
        tensor_to_grayscale_image(before_tensor).save(sample_dir / "before.png")
        tensor_to_grayscale_image(after_tensor).save(sample_dir / "after.png")
        diff_to_rgb_image(before_tensor, after_tensor).save(sample_dir / "diff.png")


def _render_clean_case(subject: tio.Subject, *, framing_mode: str, device: str) -> torch.Tensor:
    render = RenderDRR({"device": device})
    subject_copy = copy.deepcopy(subject)
    _cast_subject_images_to_float32(subject_copy)
    subject_copy = apply_framing(subject_copy, framing_mode=framing_mode)
    with torch.no_grad():
        rendered = render(subject_copy)
    return rendered["xray"].detach().cpu()


def _cast_subject_images_to_float32(subject: tio.Subject) -> None:
    for key in ("volume", "density"):
        if key in subject:
            subject[key].set_data(subject[key].data.to(torch.float32))


def _select_sample_case_ids(comparison_rows: list[dict[str, object]]) -> list[str]:
    def top(rows: list[dict[str, object]], key: str, *, reverse: bool) -> list[str]:
        candidates = [row for row in rows if row.get(key) not in (None, "")]
        ordered = sorted(candidates, key=lambda row: float(row[key]), reverse=reverse)
        return [str(row["case_id"]) for row in ordered[:10]]

    selected: list[str] = []
    selected.extend(top(comparison_rows, "before_min_edge_margin_px", reverse=False))
    selected.extend(top(comparison_rows, "after_min_edge_margin_px", reverse=False))
    selected.extend(top(comparison_rows, "delta_min_edge_margin_px", reverse=True))
    return list(dict.fromkeys(selected))


def _write_summary_markdown(  # noqa: PLR0913
    path: Path,
    *,
    before_render: dict[str, object],
    after_render: dict[str, object],
    comparison: dict[str, object],
    before_downstream: dict[str, object],
    after_downstream: dict[str, object],
    run_label: str,
) -> None:
    before_aggregate = before_downstream["aggregate"]  # type: ignore[index]
    after_aggregate = after_downstream["aggregate"]  # type: ignore[index]
    before_test_f1 = float(before_aggregate["test_f1_mean"])
    after_test_f1 = float(after_aggregate["test_f1_mean"])
    before_margin = float(before_render["min_edge_margin_mean_px"])
    after_margin = float(after_render["min_edge_margin_mean_px"])
    before_center_offset = float(before_render["center_offset_abs_mean_px"])
    after_center_offset = float(after_render["center_offset_abs_mean_px"])
    improved_render = (
        float(after_render["success_rate"]) >= float(before_render["success_rate"])
        and after_margin >= before_margin
        and after_center_offset <= before_center_offset
        and int(after_render["border_touch_count"]) <= int(before_render["border_touch_count"])
    )
    improved_downstream = after_test_f1 >= before_test_f1

    lines = [
        f"# X-ray pose benchmark: {run_label}",
        "",
        "## Verdict",
        "",
        f"- Render benchmark improved: {'yes' if improved_render else 'no'}",
        f"- Downstream test F1 improved: {'yes' if improved_downstream else 'no'}",
        "",
        "## Render",
        "",
        f"- Eligible common cases: {int(comparison['eligible_case_count'])}",
        f"- Paired successful cases: {int(comparison['paired_success_count'])}",
        f"- Before success rate: {float(before_render['success_rate']):.4f}",
        f"- After success rate: {float(after_render['success_rate']):.4f}",
        f"- Before empty count: {int(before_render['empty_count'])}",
        f"- After empty count: {int(after_render['empty_count'])}",
        f"- Before invalid-mask count: {int(before_render['invalid_mask_count'])}",
        f"- After invalid-mask count: {int(after_render['invalid_mask_count'])}",
        f"- Before mean center offset: {before_center_offset:.2f} px",
        f"- After mean center offset: {after_center_offset:.2f} px",
        f"- Before mean min-edge margin: {before_margin:.2f} px",
        f"- After mean min-edge margin: {after_margin:.2f} px",
        f"- Before median min-edge margin: {float(before_render['min_edge_margin_median_px']):.2f} px",
        f"- After median min-edge margin: {float(after_render['min_edge_margin_median_px']):.2f} px",
        f"- Before border-touch count: {int(before_render['border_touch_count'])}",
        f"- After border-touch count: {int(after_render['border_touch_count'])}",
        "",
        "## Downstream",
        "",
        f"- Before test F1 mean/std: {before_test_f1:.4f} / {float(before_aggregate['test_f1_std']):.4f}",
        f"- After test F1 mean/std: {after_test_f1:.4f} / {float(after_aggregate['test_f1_std']):.4f}",
        (
            f"- Before test accuracy mean/std: {float(before_aggregate['test_accuracy_mean']):.4f} / "
            f"{float(before_aggregate['test_accuracy_std']):.4f}"
        ),
        (
            f"- After test accuracy mean/std: {float(after_aggregate['test_accuracy_mean']):.4f} / "
            f"{float(after_aggregate['test_accuracy_std']):.4f}"
        ),
        (
            f"- Before test precision mean/std: {float(before_aggregate['test_precision_mean']):.4f} / "
            f"{float(before_aggregate['test_precision_std']):.4f}"
        ),
        (
            f"- After test precision mean/std: {float(after_aggregate['test_precision_mean']):.4f} / "
            f"{float(after_aggregate['test_precision_std']):.4f}"
        ),
        (
            f"- Before test recall mean/std: {float(before_aggregate['test_recall_mean']):.4f} / "
            f"{float(before_aggregate['test_recall_std']):.4f}"
        ),
        (
            f"- After test recall mean/std: {float(after_aggregate['test_recall_mean']):.4f} / "
            f"{float(after_aggregate['test_recall_std']):.4f}"
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
