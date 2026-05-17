from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    Swin_T_Weights,
    mobilenet_v3_large,
    resnet18,
    swin_t,
)

try:
    from .configs import (
        BASELINE_PLOT_PATH,
        CLASS_NAMES,
        DEFAULT_DATA_ROOT,
        IMAGENET_MEAN,
        IMAGENET_STD,
        RESULTS_MODELS_DIR,
        RESULTS_PLOTS_DIR,
        SUPPORTED_MODELS,
        SUPPORTED_PRIVACY_MODES,
        SUPPORTED_WEIGHTS,
        build_run_name,
        ensure_results_dirs,
    )
    from .data_loader import RAFDataset
except ImportError:
    from configs import (
        BASELINE_PLOT_PATH,
        CLASS_NAMES,
        DEFAULT_DATA_ROOT,
        IMAGENET_MEAN,
        IMAGENET_STD,
        RESULTS_MODELS_DIR,
        RESULTS_PLOTS_DIR,
        SUPPORTED_MODELS,
        SUPPORTED_PRIVACY_MODES,
        SUPPORTED_WEIGHTS,
        build_run_name,
        ensure_results_dirs,
    )
    from data_loader import RAFDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline image classifier on the RAF-style dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=SUPPORTED_MODELS,
        help="Backbone model to fine-tune.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        choices=SUPPORTED_WEIGHTS,
        help="Use pretrained ImageNet weights or random initialization.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to the dataset root containing train/val/test.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Keep 0 on Windows if needed.",
    )
    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Enable pinned host memory for faster host-to-device copies.",
    )
    pin_memory_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable pinned host memory.",
    )
    parser.set_defaults(pin_memory=None)
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs when num_workers > 0.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Number of batches prefetched by each worker when num_workers > 0.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size for the pretrained backbone.",
    )
    parser.add_argument(
        "--privacy-mode",
        type=str,
        default="none",
        choices=SUPPORTED_PRIVACY_MODES,
        help="Optional privacy filter applied inside RAFDataset.",
    )
    parser.add_argument(
        "--privacy-intensity",
        type=float,
        default=0.0,
        help="Intensity used by the selected privacy filter.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional cap for quick debugging runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=BASELINE_PLOT_PATH,
        help="Where to save the summary plot.",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to artifact filenames for experiment tracking.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print progress every N batches. Use 0 to log only epoch summaries.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs() -> tuple[Path, Path]:
    return ensure_results_dirs()


def normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower()


def build_model(model_name: str, num_classes: int, use_pretrained: bool) -> nn.Module:
    model_name = normalize_model_name(model_name)

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if use_pretrained else None
        model = mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if model_name == "swin_t":
        weights = Swin_T_Weights.DEFAULT if use_pretrained else None
        model = swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model '{model_name}'.")


class NumpyToImageNetTensor:
    def __init__(self, image_size: int, train: bool) -> None:
        augmentation = []
        if train:
            augmentation.append(transforms.RandomHorizontalFlip())

        self.pipeline = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *augmentation,
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        image = np.ascontiguousarray(image)
        pil_image = Image.fromarray(image.astype(np.uint8))
        return self.pipeline(pil_image)


def build_transform(image_size: int, train: bool) -> Callable[[np.ndarray], torch.Tensor]:
    return NumpyToImageNetTensor(image_size=image_size, train=train)


def maybe_limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, range(max_samples))


def resolve_pin_memory(args: argparse.Namespace) -> bool:
    if args.pin_memory is None:
        return torch.cuda.is_available()
    return args.pin_memory


def resolve_loader_kwargs(args: argparse.Namespace, shuffle: bool) -> dict[str, object]:
    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": resolve_pin_memory(args),
    }

    if args.num_workers > 0 and args.persistent_workers:
        loader_kwargs["persistent_workers"] = True

    if args.num_workers > 0 and args.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    return loader_kwargs


def create_dataloaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], list[str]]:
    train_dataset = RAFDataset(
        root_dir=args.data_root,
        split="train",
        mode=args.privacy_mode,
        intensity=args.privacy_intensity,
        transform=build_transform(args.image_size, train=True),
        grayscale=True,
    )
    val_dataset = RAFDataset(
        root_dir=args.data_root,
        split="val",
        mode=args.privacy_mode,
        intensity=args.privacy_intensity,
        transform=build_transform(args.image_size, train=False),
        grayscale=True,
    )
    test_dataset = RAFDataset(
        root_dir=args.data_root,
        split="test",
        mode=args.privacy_mode,
        intensity=args.privacy_intensity,
        transform=build_transform(args.image_size, train=False),
        grayscale=True,
    )

    train_dataset = maybe_limit_dataset(train_dataset, args.max_samples_per_split)
    val_dataset = maybe_limit_dataset(val_dataset, args.max_samples_per_split)
    test_dataset = maybe_limit_dataset(test_dataset, args.max_samples_per_split)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            **resolve_loader_kwargs(args, shuffle=True),
        ),
        "val": DataLoader(
            val_dataset,
            **resolve_loader_kwargs(args, shuffle=False),
        ),
        "test": DataLoader(
            test_dataset,
            **resolve_loader_kwargs(args, shuffle=False),
        ),
    }

    class_names = list(CLASS_NAMES)
    return dataloaders, class_names


def compute_metrics(
    targets: list[int],
    predictions: list[int],
    num_classes: int,
) -> dict[str, float]:
    labels = list(range(num_classes))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    split_name: str = "train",
    log_interval: int = 50,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    non_blocking = device.type == "cuda"

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    split_start = time.time()
    total_batches = len(dataloader)
    total_samples = len(dataloader.dataset)

    for batch_index, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)

        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

        should_log = (
            log_interval > 0
            and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches)
        )
        if should_log:
            processed_samples = min(batch_index * dataloader.batch_size, total_samples)
            average_loss = total_loss / processed_samples
            elapsed_minutes = (time.time() - split_start) / 60
            epoch_label = (
                f"epoch={epoch:02d}/{total_epochs}"
                if epoch is not None and total_epochs is not None
                else "epoch=?"
            )
            print(
                f"{split_name:<5} | {epoch_label} | "
                f"batch={batch_index:04d}/{total_batches} | "
                f"samples={processed_samples}/{total_samples} | "
                f"avg_loss={average_loss:.4f} | "
                f"elapsed={elapsed_minutes:.1f}m",
                flush=True,
            )

    metrics = compute_metrics(all_targets, all_predictions, num_classes=num_classes)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics


def evaluate_on_test(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[dict[str, float], np.ndarray, str]:
    model.eval()
    non_blocking = device.type == "cuda"
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    labels = list(range(len(class_names)))

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            logits = model(images)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics = compute_metrics(
        all_targets,
        all_predictions,
        num_classes=len(class_names),
    )
    metrics["loss"] = total_loss / len(dataloader.dataset)

    report = classification_report(
        all_targets,
        all_predictions,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    matrix = confusion_matrix(all_targets, all_predictions, labels=labels)
    return metrics, matrix, report


def plot_results(
    history: dict[str, list[dict[str, float]]],
    test_metrics: dict[str, float],
    confusion: np.ndarray,
    class_names: list[str],
    plot_path: Path,
    title_suffix: str,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    train_losses = [epoch["loss"] for epoch in history["train"]]
    val_losses = [epoch["loss"] for epoch in history["val"]]
    axes[0, 0].plot(epochs, train_losses, marker="o", label="Train")
    axes[0, 0].plot(epochs, val_losses, marker="o", label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-entropy loss")
    axes[0, 0].legend()

    train_accuracy = [epoch["accuracy"] for epoch in history["train"]]
    val_accuracy = [epoch["accuracy"] for epoch in history["val"]]
    axes[0, 1].plot(epochs, train_accuracy, marker="o", label="Train")
    axes[0, 1].plot(epochs, val_accuracy, marker="o", label="Validation")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]
    metric_values = [test_metrics[name] for name in metric_names]
    axes[1, 0].bar(metric_names, metric_values, color="#355070")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("Test Metrics")
    axes[1, 0].tick_params(axis="x", rotation=25)

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized_confusion = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_sums != 0,
    )
    heatmap = axes[1, 1].imshow(normalized_confusion, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Normalized Confusion Matrix")
    axes[1, 1].set_xticks(range(len(class_names)))
    axes[1, 1].set_yticks(range(len(class_names)))
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1, 1].set_yticklabels(class_names)
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("True")

    for row in range(normalized_confusion.shape[0]):
        for col in range(normalized_confusion.shape[1]):
            value = normalized_confusion[row, col]
            axes[1, 1].text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )

    fig.colorbar(heatmap, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Baseline Training Summary - {title_suffix}", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_artifacts(
    model: nn.Module,
    args: argparse.Namespace,
    history: dict[str, list[dict[str, float]]],
    test_metrics: dict[str, float],
    test_report: str,
    class_names: list[str],
    best_epoch: int,
) -> tuple[Path, Path]:
    RESULTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = build_run_name(
        normalize_model_name(args.model),
        args.privacy_mode,
        args.privacy_intensity,
        args.run_suffix,
    )

    checkpoint_path = RESULTS_MODELS_DIR / f"{run_name}_best.pt"
    metrics_path = RESULTS_MODELS_DIR / f"{run_name}_metrics.json"
    report_path = RESULTS_MODELS_DIR / f"{run_name}_classification_report.txt"

    torch.save(
        {
            "model_name": normalize_model_name(args.model),
            "weights": args.weights,
            "privacy_mode": args.privacy_mode,
            "privacy_intensity": args.privacy_intensity,
            "run_name": run_name,
            "run_suffix": args.run_suffix,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "pin_memory": resolve_pin_memory(args),
            "persistent_workers": bool(args.num_workers > 0 and args.persistent_workers),
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "history": history,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )

    metrics_payload = {
        "model": normalize_model_name(args.model),
        "weights": args.weights,
        "privacy_mode": args.privacy_mode,
        "privacy_intensity": args.privacy_intensity,
        "run_name": run_name,
        "run_suffix": args.run_suffix,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "pin_memory": resolve_pin_memory(args),
        "persistent_workers": bool(args.num_workers > 0 and args.persistent_workers),
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "history": history,
        "test_metrics": test_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    report_path.write_text(test_report, encoding="utf-8")
    return checkpoint_path, metrics_path


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_output_dirs()

    dataloaders, class_names = create_dataloaders(args)
    num_classes = len(class_names)
    use_pretrained = args.weights == "pretrained"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=num_classes, use_pretrained=use_pretrained)
    model = model.to(device)

    print(
        "DataLoader config: "
        f"batch_size={args.batch_size} | "
        f"num_workers={args.num_workers} | "
        f"pin_memory={resolve_pin_memory(args)} | "
        f"persistent_workers={bool(args.num_workers > 0 and args.persistent_workers)} | "
        f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else None}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = {"train": [], "val": []}
    best_val_f1 = -1.0
    best_epoch = 0
    best_state_dict = None

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            split_name="train",
            log_interval=args.log_interval,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            optimizer=None,
            epoch=epoch,
            total_epochs=args.epochs,
            split_name="val",
            log_interval=args.log_interval,
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state_dict)

    test_metrics, confusion, report = evaluate_on_test(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=device,
        class_names=class_names,
    )

    checkpoint_path, metrics_path = save_training_artifacts(
        model=model,
        args=args,
        history=history,
        test_metrics=test_metrics,
        test_report=report,
        class_names=class_names,
        best_epoch=best_epoch,
    )

    title_suffix = (
        f"{normalize_model_name(args.model)} | "
        f"{args.privacy_mode} ({args.privacy_intensity})"
    )
    plot_results(
        history=history,
        test_metrics=test_metrics,
        confusion=confusion,
        class_names=class_names,
        plot_path=args.plot_path,
        title_suffix=title_suffix,
    )

    elapsed_seconds = time.time() - start_time
    print()
    print("Training finished.")
    print(f"Device: {device}")
    print(f"Best epoch: {best_epoch}")
    print(f"Plot saved to: {args.plot_path}")
    print(f"Model checkpoint saved to: {checkpoint_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Elapsed time: {elapsed_seconds / 60:.2f} minutes")
    print()
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print()
    print("Classification report:")
    print(report)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
