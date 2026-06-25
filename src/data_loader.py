from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .configs import DEFAULT_DATA_ROOT
    from .privacy_filters import (
        apply_canny_edges,
        apply_center_crop,
        apply_diffusion_noise,
        apply_gaussian_blur,
        apply_mosaic,
    )
except ImportError:
    from configs import DEFAULT_DATA_ROOT
    from privacy_filters import (
        apply_canny_edges,
        apply_center_crop,
        apply_diffusion_noise,
        apply_gaussian_blur,
        apply_mosaic,
    )

SUPPORTED_SPLITS = ("train", "val", "test")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ANNOTATION_CANDIDATES = (
    "list_pat_rev_label.txt",
    "list_patition_label.txt",
    "EmoLabel/list_pat_rev_label.txt",
    "EmoLabel/list_patition_label.txt",
)
IMAGE_DIR_CANDIDATES = (
    "Image/aligned",
    "Image/original",
    "images",
    ".",
)


# Official RAF-DB labels are 1-based.
RAF_LABEL_TO_CLASS = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral",
}

CLASS_NAME_ALIASES = {
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "angry": "angry",
    "anger": "angry",
    "neutral": "neutral",
}

CLASS_TO_RAF_LABEL = {
    class_name: label for label, class_name in RAF_LABEL_TO_CLASS.items()
}


@dataclass(frozen=True)
class RAFRecord:
    image_path: Path
    file_name: str
    split: str
    class_name: str
    raf_label: int
    target: int


class RAFDataset(Dataset):
    """
    Custom PyTorch dataset for RAF-DB style data.

    Supports both:
    1. The original RAF-DB annotation file format, such as `list_pat_rev_label.txt`.
    2. A folder-based layout like `root/train/happy/*.jpg`.

    Example:
        dataset = RAFDataset(split="train", mode="blur", intensity=5)
    """

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_DATA_ROOT,
        split: str = "train",
        labels_file: str | Path | None = None,
        images_dir: str | Path | None = None,
        mode: str | None = None,
        intensity: float = 0.0,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        grayscale: bool = True,
        return_metadata: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.labels_file = self._resolve_optional_path(labels_file)
        self.images_dir = self._resolve_optional_path(images_dir)
        self.mode = self._normalize_privacy_mode(mode)
        self.intensity = float(intensity)
        self.transform = transform
        self.target_transform = target_transform
        self.grayscale = grayscale
        self.return_metadata = return_metadata
        self.selected_splits = self._resolve_splits(self.split)
        self.classes = [RAF_LABEL_TO_CLASS[index] for index in sorted(RAF_LABEL_TO_CLASS)]
        self.class_to_idx = {
            class_name: label - 1 for label, class_name in RAF_LABEL_TO_CLASS.items()
        }

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        resolved_labels_file = self._resolve_labels_file()
        if resolved_labels_file is not None:
            self.records = self._load_records_from_annotations(resolved_labels_file)
            self.data_source = "annotations"
        else:
            self.records = self._load_records_from_folders()
            self.data_source = "folders"

        if not self.records:
            raise ValueError(
                f"No samples found for split='{self.split}' in {self.root_dir}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = self._load_image(record.image_path)
        image = self._apply_privacy_filter(image)
        image_tensor = self._apply_transform(image)

        target = record.target
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_metadata:
            return {
                "image": image_tensor,
                "target": target,
                "class_name": record.class_name,
                "raf_label": record.raf_label,
                "file_name": record.file_name,
                "image_path": str(record.image_path),
                "split": record.split,
            }

        return image_tensor, target

    def _resolve_optional_path(self, path: str | Path | None) -> Path | None:
        if path is None:
            return None

        path = Path(path)
        if path.is_absolute():
            return path

        return self.root_dir / path

    def _resolve_splits(self, split: str) -> tuple[str, ...]:
        if split in SUPPORTED_SPLITS:
            return (split,)
        if split in {"all", "full"}:
            return SUPPORTED_SPLITS
        raise ValueError(
            f"Unsupported split '{split}'. Expected one of {SUPPORTED_SPLITS} or 'all'."
        )

    def _normalize_privacy_mode(self, mode: str | None) -> str:
        if mode is None:
            return "none"

        normalized = mode.strip().lower()
        mode_aliases = {
            "none": "none",
            "original": "none",
            "blur": "blur",
            "gaussian": "blur",
            "gaussian_blur": "blur",
            "edges": "edges",
            "canny": "edges",
            "canny_edges": "edges",
            "crop": "crop",
            "center_crop": "crop",
            "central_crop": "crop",
            "mosaic": "mosaic",
            "pixelate": "mosaic",
            "pixelation": "mosaic",
            "noise": "noise",
            "diffusion": "noise",
            "diffusion_noise": "noise",
        }

        if normalized not in mode_aliases:
            raise ValueError(
                "Unsupported privacy mode. Use one of: none, blur, crop, mosaic, edges, noise."
            )

        return mode_aliases[normalized]

    def _resolve_labels_file(self) -> Path | None:
        if self.labels_file is not None:
            if not self.labels_file.exists():
                raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
            return self.labels_file

        for candidate in ANNOTATION_CANDIDATES:
            candidate_path = self.root_dir / candidate
            if candidate_path.exists():
                return candidate_path

        return None

    def _load_records_from_annotations(self, labels_file: Path) -> list[RAFRecord]:
        search_roots = self._resolve_image_search_roots()
        image_lookup = self._build_image_lookup(search_roots)
        records: list[RAFRecord] = []

        with labels_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                parts = re.split(r"[\s,]+", stripped)
                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid label format in {labels_file} at line {line_number}: {stripped}"
                    )

                image_name = Path(parts[0]).name
                split = self._infer_split(parts[0])
                if split not in self.selected_splits:
                    continue

                raf_label = int(parts[1])
                class_name = RAF_LABEL_TO_CLASS.get(raf_label)
                if class_name is None:
                    raise ValueError(
                        f"Unknown RAF label '{raf_label}' in {labels_file} at line {line_number}."
                    )

                image_path = self._resolve_image_path(image_name, image_lookup)
                records.append(
                    RAFRecord(
                        image_path=image_path,
                        file_name=image_name,
                        split=split,
                        class_name=class_name,
                        raf_label=raf_label,
                        target=raf_label - 1,
                    )
                )

        return records

    def _resolve_image_search_roots(self) -> list[Path]:
        if self.images_dir is not None:
            if not self.images_dir.exists():
                raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
            return [self.images_dir]

        search_roots = []
        for candidate in IMAGE_DIR_CANDIDATES:
            candidate_path = self.root_dir / candidate
            if candidate_path.exists():
                search_roots.append(candidate_path)

        if not search_roots:
            search_roots.append(self.root_dir)

        return search_roots

    def _build_image_lookup(self, search_roots: list[Path]) -> dict[str, Path]:
        image_lookup: dict[str, Path] = {}

        for search_root in search_roots:
            for path in search_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_lookup[path.name] = path

        return image_lookup

    def _resolve_image_path(self, image_name: str, image_lookup: dict[str, Path]) -> Path:
        image_path = image_lookup.get(image_name)
        if image_path is not None:
            return image_path

        stem = Path(image_name).stem
        suffix = Path(image_name).suffix
        aligned_name = f"{stem}_aligned{suffix}"
        image_path = image_lookup.get(aligned_name)
        if image_path is not None:
            return image_path

        raise FileNotFoundError(
            f"Could not resolve image '{image_name}' from the annotation file."
        )

    def _load_records_from_folders(self) -> list[RAFRecord]:
        records: list[RAFRecord] = []

        for split in self.selected_splits:
            split_dir = self.root_dir / split
            if not split_dir.exists():
                continue

            class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
            for class_dir in class_dirs:
                class_name = self._normalize_class_name(class_dir.name)
                raf_label = CLASS_TO_RAF_LABEL[class_name]

                image_paths = sorted(
                    path
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )

                for image_path in image_paths:
                    records.append(
                        RAFRecord(
                            image_path=image_path,
                            file_name=image_path.name,
                            split=split,
                            class_name=class_name,
                            raf_label=raf_label,
                            target=raf_label - 1,
                        )
                    )

        return records

    def _normalize_class_name(self, class_name: str) -> str:
        normalized = class_name.strip().lower()
        if normalized not in CLASS_NAME_ALIASES:
            raise ValueError(f"Unsupported class name '{class_name}' in dataset folders.")
        return CLASS_NAME_ALIASES[normalized]

    def _infer_split(self, image_reference: str) -> str:
        normalized = image_reference.replace("\\", "/").lower()
        path_parts = normalized.split("/")

        for split in SUPPORTED_SPLITS:
            if split in path_parts or Path(normalized).name.startswith(f"{split}_"):
                return split

        if len(self.selected_splits) == 1:
            return self.selected_splits[0]

        raise ValueError(
            f"Could not infer split from annotation entry '{image_reference}'."
        )

    def _load_image(self, image_path: Path) -> np.ndarray:
        read_flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(image_path), read_flag)

        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        if not self.grayscale and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _apply_privacy_filter(self, image: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return image
        if self.mode == "blur":
            return apply_gaussian_blur(image, self.intensity)
        if self.mode == "crop":
            return apply_center_crop(image, self.intensity)
        if self.mode == "mosaic":
            return apply_mosaic(image, self.intensity)
        if self.mode == "edges":
            return apply_canny_edges(image)
        if self.mode == "noise":
            return apply_diffusion_noise(image, self.intensity)
        raise RuntimeError(f"Unhandled privacy mode: {self.mode}")

    def _apply_transform(self, image: np.ndarray) -> torch.Tensor:
        transformed = self.transform(image) if self.transform is not None else image

        if torch.is_tensor(transformed):
            return transformed

        if not isinstance(transformed, np.ndarray):
            raise TypeError(
                "Transform output must be a torch.Tensor or numpy.ndarray."
            )

        array = np.ascontiguousarray(transformed)
        tensor = torch.from_numpy(array)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError("Image tensor must have 2 or 3 dimensions.")

        if np.issubdtype(array.dtype, np.integer):
            max_value = float(np.iinfo(array.dtype).max)
            return tensor.float() / max_value

        tensor = tensor.float()
        if torch.max(tensor) > 1.0:
            tensor = tensor / 255.0

        return tensor

    def get_class_distribution(self) -> dict[str, int]:
        distribution = {class_name: 0 for class_name in self.classes}
        for record in self.records:
            distribution[record.class_name] += 1
        return distribution


def create_data_loader(
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    **dataset_kwargs,
) -> DataLoader:
    dataset = RAFDataset(split=split, **dataset_kwargs)

    if shuffle is None:
        shuffle = split == "train"
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
