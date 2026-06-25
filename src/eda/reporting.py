from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from ..configs import CLASS_NAMES, DEFAULT_DATA_ROOT
from ..data.loader import SUPPORTED_EXTENSIONS


SPLITS = ("train", "val", "test")


def collect_dataset_inventory(data_root: Path = DEFAULT_DATA_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")

    structure_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []

    for split in SPLITS:
        split_path = data_root / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split directory: {split_path}")

        observed_classes = sorted(path.name for path in split_path.iterdir() if path.is_dir())
        missing_classes = [name for name in CLASS_NAMES if name not in observed_classes]
        unexpected_classes = [name for name in observed_classes if name not in CLASS_NAMES]

        split_total = 0
        for class_name in CLASS_NAMES:
            class_dir = split_path / class_name
            image_paths = []
            if class_dir.exists():
                image_paths = sorted(
                    path
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )

            split_total += len(image_paths)
            for image_path in image_paths:
                image_rows.append(
                    {
                        "split": split,
                        "label": class_name,
                        "filename": image_path.name,
                        "extension": image_path.suffix.lower(),
                        "image_path": str(image_path),
                    }
                )

        structure_rows.append(
            {
                "split": split,
                "num_expected_classes_found": len(
                    [name for name in CLASS_NAMES if (split_path / name).exists()]
                ),
                "missing_classes": ", ".join(missing_classes) if missing_classes else "none",
                "unexpected_classes": ", ".join(unexpected_classes) if unexpected_classes else "none",
                "num_images": split_total,
            }
        )

    return pd.DataFrame(image_rows), pd.DataFrame(structure_rows)


def class_count_tables(images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_counts = (
        images.groupby(["label", "split"])
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = (
        class_counts.pivot(index="label", columns="split", values="count")
        .reindex(index=CLASS_NAMES, columns=SPLITS)
        .fillna(0)
        .astype(int)
    )
    balance_summary = pd.DataFrame(
        {
            "split": list(SPLITS),
            "total_images": [int(pivot[split].sum()) for split in SPLITS],
            "min_class_count": [int(pivot[split].min()) for split in SPLITS],
            "max_class_count": [int(pivot[split].max()) for split in SPLITS],
            "balanced_within_split": [
                bool(pivot[split].nunique() == 1) for split in SPLITS
            ],
        }
    )
    return pivot, balance_summary


def plot_dataset_balance(images: pd.DataFrame, class_counts: pd.DataFrame) -> plt.Figure:
    split_counts = images["split"].value_counts().reindex(SPLITS).fillna(0).astype(int)
    colors = ["#355070", "#6d597a", "#b56576"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    split_counts.plot(kind="bar", ax=axes[0], color=colors)
    axes[0].set_title("Images per split")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Number of images")
    axes[0].tick_params(axis="x", rotation=0)

    class_counts.plot(kind="bar", ax=axes[1], color=colors)
    axes[1].set_title("Images per class and split")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Number of images")
    axes[1].legend(title="Split")
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    return fig


def inspect_image_properties(
    images: pd.DataFrame,
    sample_size: int = 1000,
    random_seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    file_extensions = images["extension"].value_counts().rename("count")
    sample = images.sample(n=min(len(images), sample_size), random_state=random_seed)
    rows: list[dict[str, object]] = []

    for image_path in sample["image_path"]:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            rows.append(
                {
                    "read_ok": False,
                    "height": None,
                    "width": None,
                    "channels": None,
                    "dtype": None,
                }
            )
            continue

        if image.ndim == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape

        rows.append(
            {
                "read_ok": True,
                "height": height,
                "width": width,
                "channels": channels,
                "dtype": str(image.dtype),
            }
        )

    return file_extensions, pd.DataFrame(rows)


def plot_raw_samples(
    images: pd.DataFrame,
    split: str = "train",
    samples_per_class: int = 3,
    random_seed: int = 42,
) -> plt.Figure:
    fig, axes = plt.subplots(
        len(CLASS_NAMES),
        samples_per_class,
        figsize=(3.2 * samples_per_class, 2.4 * len(CLASS_NAMES)),
    )

    for row, class_name in enumerate(CLASS_NAMES):
        subset = images[(images["split"] == split) & (images["label"] == class_name)]
        sample_count = min(samples_per_class, len(subset))
        sample_df = subset.sample(n=sample_count, random_state=random_seed)

        for col in range(samples_per_class):
            axis = axes[row, col]
            if col < sample_count:
                image_path = Path(sample_df.iloc[col]["image_path"])
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                axis.imshow(image, cmap="gray")
                axis.set_title(f"{class_name}\n{image_path.name[:18]}", fontsize=9)
            axis.axis("off")

    fig.suptitle(f"Raw samples from the {split} split", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig
