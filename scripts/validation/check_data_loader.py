from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configs import CLASS_NAMES, DEFAULT_DATA_ROOT
from src.data.loader import RAFDataset, create_data_loader


SPLITS = ("train", "val", "test")
PRIVACY_CHECKS = (
    ("original", "none", 0.0),
    ("crop_context", "crop", 0.75),
    ("blur", "blur", 3.0),
    ("mosaic", "mosaic", 8.0),
    ("canny", "edges", 0.0),
    ("noise", "noise", 100.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick sanity check for the RAF-style dataset and DataLoader."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing train/val/test folders.",
    )
    return parser.parse_args()


def check_dataset_root(data_root: Path) -> None:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    print(f"Dataset root: {data_root}")


def check_split_sizes(data_root: Path) -> None:
    print("\nSplit sizes:")
    for split in SPLITS:
        dataset = RAFDataset(root_dir=data_root, split=split)
        distribution = dataset.get_class_distribution()
        is_balanced = len(set(distribution.values())) == 1

        print(
            f"- {split}: {len(dataset):,} samples | "
            f"{len(dataset.classes)} classes | balanced={is_balanced}"
        )

        if dataset.classes != list(CLASS_NAMES):
            raise AssertionError(
                f"Class order mismatch in split '{split}': {dataset.classes}"
            )


def check_metadata_and_filters(data_root: Path) -> None:
    print("\nPrivacy filter sanity check:")
    baseline = RAFDataset(
        root_dir=data_root,
        split="train",
        mode="none",
        return_metadata=True,
    )
    reference = baseline[0]
    reference_file = reference["file_name"]

    print(
        f"- reference sample: {reference_file} | "
        f"class={reference['class_name']} | target={reference['target']}"
    )

    for label, mode, intensity in PRIVACY_CHECKS:
        dataset = RAFDataset(
            root_dir=data_root,
            split="train",
            mode=mode,
            intensity=intensity,
            return_metadata=True,
        )
        sample = dataset[0]
        image = sample["image"]

        if sample["file_name"] != reference_file:
            raise AssertionError(
                f"Filter '{label}' did not use the same source sample."
            )

        print(
            f"- {label}: shape={tuple(image.shape)} | "
            f"dtype={image.dtype} | min={float(image.min()):.3f} | "
            f"max={float(image.max()):.3f}"
        )


def check_batch_loading(data_root: Path) -> None:
    print("\nDataLoader batch check:")
    loader = create_data_loader(
        root_dir=data_root,
        split="train",
        batch_size=8,
        mode="crop",
        intensity=0.75,
        shuffle=True,
        num_workers=0,
    )
    images, targets = next(iter(loader))

    if images.ndim != 4:
        raise AssertionError(f"Expected image batch with 4 dims, got {images.shape}.")
    if targets.ndim != 1:
        raise AssertionError(f"Expected target batch with 1 dim, got {targets.shape}.")
    if images.dtype != torch.float32:
        raise AssertionError(f"Expected image dtype torch.float32, got {images.dtype}.")

    print(f"- images: shape={tuple(images.shape)} | dtype={images.dtype}")
    print(f"- targets: shape={tuple(targets.shape)} | dtype={targets.dtype}")
    print(f"- unique targets: {sorted(targets.unique().tolist())}")


def main() -> None:
    args = parse_args()
    check_dataset_root(args.data_root)
    check_split_sizes(args.data_root)
    check_metadata_and_filters(args.data_root)
    check_batch_loading(args.data_root)
    print("\nDataLoader sanity check passed.")


if __name__ == "__main__":
    main()
