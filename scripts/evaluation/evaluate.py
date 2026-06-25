from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import (
    CLASS_NAMES,
    DEFAULT_DATA_ROOT,
    RESULTS_EVALUATIONS_DIR,
    RESULTS_TABLES_DIR,
    SUPPORTED_MODELS,
    SUPPORTED_PRIVACY_MODES,
    ensure_results_dirs,
)
from src.data.loader import RAFDataset
from src.modeling.training import (
    build_model,
    build_transform,
    evaluate_on_test,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained FER checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a *_best.pt checkpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=SUPPORTED_MODELS,
        help="Model name. If omitted, read from checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing train/val/test.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--privacy-mode",
        type=str,
        default=None,
        choices=SUPPORTED_PRIVACY_MODES,
        help="Privacy mode. If omitted, read from checkpoint.",
    )
    parser.add_argument(
        "--privacy-intensity",
        type=float,
        default=None,
        help="Privacy intensity. If omitted, read from checkpoint.",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_EVALUATIONS_DIR,
        help="Directory for evaluation artifacts.",
    )
    return parser.parse_args()


def _checkpoint_run_name(checkpoint: dict, checkpoint_path: Path) -> str:
    if checkpoint.get("run_name"):
        return str(checkpoint["run_name"])
    return checkpoint_path.stem.replace("_best", "")


def _checkpoint_model_name(checkpoint: dict, args: argparse.Namespace) -> str:
    model_name = args.model or checkpoint.get("model_name") or checkpoint.get("model")
    if not model_name:
        raise ValueError("Model name was not found in checkpoint. Pass --model.")
    return str(model_name)


def _checkpoint_privacy_mode(checkpoint: dict, args: argparse.Namespace) -> str:
    return str(args.privacy_mode or checkpoint.get("privacy_mode") or "none")


def _checkpoint_privacy_intensity(checkpoint: dict, args: argparse.Namespace) -> float:
    if args.privacy_intensity is not None:
        return float(args.privacy_intensity)
    return float(checkpoint.get("privacy_intensity", 0.0))


def _checkpoint_image_size(checkpoint: dict, args: argparse.Namespace) -> int:
    if args.image_size is not None:
        return int(args.image_size)
    return int(checkpoint.get("image_size", 224))


def create_loader(
    args: argparse.Namespace,
    privacy_mode: str,
    privacy_intensity: float,
    image_size: int,
) -> DataLoader:
    dataset = RAFDataset(
        root_dir=args.data_root,
        split=args.split,
        mode=privacy_mode,
        intensity=privacy_intensity,
        transform=build_transform(image_size=image_size, train=False),
        grayscale=True,
    )
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = Subset(dataset, range(args.max_samples))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def append_evaluation_summary(row: dict[str, object]) -> Path:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_TABLES_DIR / "evaluation_runs.csv"
    row_df = pd.DataFrame([row])
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path)
        output_df = pd.concat([existing_df, row_df], ignore_index=True)
    else:
        output_df = row_df
    output_df.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_results_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_name = _checkpoint_run_name(checkpoint, args.checkpoint)
    model_name = _checkpoint_model_name(checkpoint, args)
    privacy_mode = _checkpoint_privacy_mode(checkpoint, args)
    privacy_intensity = _checkpoint_privacy_intensity(checkpoint, args)
    image_size = _checkpoint_image_size(checkpoint, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=model_name,
        num_classes=len(CLASS_NAMES),
        use_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loader = create_loader(
        args=args,
        privacy_mode=privacy_mode,
        privacy_intensity=privacy_intensity,
        image_size=image_size,
    )
    criterion = nn.CrossEntropyLoss()
    metrics, confusion, report = evaluate_on_test(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        class_names=list(CLASS_NAMES),
    )

    artifact_stem = f"{run_name}_{args.split}_evaluation"
    metrics_path = args.output_dir / f"{artifact_stem}_metrics.json"
    report_path = args.output_dir / f"{artifact_stem}_classification_report.txt"
    confusion_path = args.output_dir / f"{artifact_stem}_confusion_matrix.csv"

    payload = {
        "run_name": run_name,
        "checkpoint_path": str(args.checkpoint),
        "model": model_name,
        "split": args.split,
        "data_root": str(args.data_root),
        "privacy_mode": privacy_mode,
        "privacy_intensity": privacy_intensity,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "metrics": metrics,
        "confusion_matrix": confusion.tolist(),
        "class_names": list(CLASS_NAMES),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    pd.DataFrame(confusion, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(confusion_path)

    summary_row = {
        "run_name": run_name,
        "model": model_name,
        "split": args.split,
        "privacy_mode": privacy_mode,
        "privacy_intensity": privacy_intensity,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "loss": metrics["loss"],
        "max_samples": args.max_samples,
        "checkpoint_path": str(args.checkpoint),
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(confusion_path),
        "classification_report_path": str(report_path),
    }
    summary_path = append_evaluation_summary(summary_row)

    print("Evaluation finished.")
    print(f"Device: {device}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Report saved to: {report_path}")
    print(f"Confusion matrix saved to: {confusion_path}")
    print(f"Summary table updated: {summary_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
