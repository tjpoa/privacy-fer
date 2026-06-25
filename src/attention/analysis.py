from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

try:
    from ..configs import CLASS_NAMES, DEFAULT_DATA_ROOT
    from ..data.loader import RAFDataset
    from ..privacy.filters import apply_center_crop, apply_gaussian_blur, apply_mosaic
    from ..modeling.training import NumpyToImageNetTensor, build_model
except ImportError:
    from src.configs import CLASS_NAMES, DEFAULT_DATA_ROOT
    from src.data.loader import RAFDataset
    from src.privacy.filters import apply_center_crop, apply_gaussian_blur, apply_mosaic
    from src.modeling.training import NumpyToImageNetTensor, build_model


@dataclass(frozen=True)
class AttentionCondition:
    label: str
    mode: str
    intensity: float
    config: object


def read_grayscale_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def apply_condition_filter(
    image: np.ndarray,
    mode: str,
    intensity: float,
) -> np.ndarray:
    if mode == "none":
        return image.copy()
    if mode == "blur":
        return apply_gaussian_blur(image, intensity)
    if mode == "mosaic":
        return apply_mosaic(image, intensity)
    if mode == "crop":
        return apply_center_crop(image, intensity)
    raise ValueError(f"Unsupported attention condition mode: {mode}")


def load_checkpoint_model(config: object, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(
        config.checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = build_model(
        config.model,
        num_classes=len(CLASS_NAMES),
        use_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def find_last_vit_block(model: torch.nn.Module) -> torch.nn.Module:
    return list(model.encoder.layers.children())[-1]


def vit_attention_map(
    model: torch.nn.Module,
    image: np.ndarray,
    image_size: int = 224,
) -> tuple[np.ndarray, int]:
    transform = NumpyToImageNetTensor(image_size=image_size, train=False)
    device = next(model.parameters()).device
    captured: dict[str, torch.Tensor] = {}
    block = find_last_vit_block(model)

    def capture_attention_input(module, inputs):
        captured["x"] = inputs[0].detach()

    handle = block.self_attention.register_forward_pre_hook(capture_attention_input)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prediction = int(logits.argmax(dim=1).item())

    handle.remove()
    if "x" not in captured:
        raise RuntimeError("Could not capture ViT attention input.")

    with torch.no_grad():
        _, weights = block.self_attention(
            captured["x"],
            captured["x"],
            captured["x"],
            need_weights=True,
            average_attn_weights=False,
        )

    attention = weights[0].mean(dim=0)[0, 1:]
    grid_size = int(math.sqrt(attention.numel()))
    attention = attention.reshape(grid_size, grid_size).cpu().numpy()
    attention = cv2.resize(
        attention,
        (image_size, image_size),
        interpolation=cv2.INTER_CUBIC,
    )
    attention = attention - attention.min()
    if attention.max() > 0:
        attention = attention / attention.max()
    return attention, prediction


def predict_image(
    model: torch.nn.Module,
    image: np.ndarray,
    image_size: int = 224,
) -> tuple[int, float, np.ndarray]:
    transform = NumpyToImageNetTensor(image_size=image_size, train=False)
    device = next(model.parameters()).device
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    prediction = int(probabilities.argmax().item())
    confidence = float(probabilities[prediction].item())
    return prediction, confidence, probabilities.cpu().numpy()


def attention_region_mask(
    attention: np.ndarray,
    mask_fraction: float = 0.2,
    region: str = "high",
) -> np.ndarray:
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between 0 and 1.")

    values = np.asarray(attention, dtype=np.float32)
    if region == "high":
        threshold = np.quantile(values, 1.0 - mask_fraction)
        return values >= threshold
    if region == "low":
        threshold = np.quantile(values, mask_fraction)
        return values <= threshold
    raise ValueError("region must be 'high' or 'low'.")


def apply_attention_mask(
    image: np.ndarray,
    attention: np.ndarray,
    mask_fraction: float = 0.2,
    region: str = "high",
    image_size: int = 224,
    fill_value: float | None = None,
) -> np.ndarray:
    resized = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    mask = attention_region_mask(
        attention=attention,
        mask_fraction=mask_fraction,
        region=region,
    )
    if mask.shape != resized.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (resized.shape[1], resized.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    masked = resized.copy()
    fill = float(np.mean(resized)) if fill_value is None else float(fill_value)
    masked[mask] = np.clip(fill, 0, 255)
    return masked.astype(np.uint8)


def overlay_attention(
    image: np.ndarray,
    attention: np.ndarray,
    image_size: int = 224,
) -> np.ndarray:
    image_resized = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    if image_resized.ndim == 2:
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_resized

    heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (0.55 * image_rgb + 0.45 * heatmap).clip(0, 255).astype(np.uint8)


def attention_entropy(attention: np.ndarray, eps: float = 1e-12) -> float:
    values = np.asarray(attention, dtype=np.float64).ravel()
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total <= eps:
        return 0.0

    probabilities = values / total
    entropy = -np.sum(probabilities * np.log(probabilities + eps))
    return float(entropy / np.log(probabilities.size))


def attention_cosine_similarity(
    attention: np.ndarray,
    reference_attention: np.ndarray,
    eps: float = 1e-12,
) -> float:
    values = np.asarray(attention, dtype=np.float64).ravel()
    reference = np.asarray(reference_attention, dtype=np.float64).ravel()
    denominator = np.linalg.norm(values) * np.linalg.norm(reference)
    if denominator <= eps:
        return 0.0
    return float(np.dot(values, reference) / denominator)


def collect_condition_predictions(
    condition: AttentionCondition,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 0,
    max_samples: int | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(condition.config, device=device)

    dataset = RAFDataset(
        root_dir=data_root,
        split="test",
        mode=condition.mode,
        intensity=condition.intensity,
        transform=NumpyToImageNetTensor(image_size=image_size, train=False),
        grayscale=True,
        return_metadata=True,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            targets = batch["target"]
            if torch.is_tensor(targets):
                target_values = targets.cpu().tolist()
            else:
                target_values = list(targets)

            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().tolist()

            for index, prediction in enumerate(predictions):
                true_index = int(target_values[index])
                rows.append(
                    {
                        "condition": condition.label,
                        "mode": condition.mode,
                        "intensity": condition.intensity,
                        "image_path": batch["image_path"][index],
                        "file_name": batch["file_name"][index],
                        "true_class": batch["class_name"][index],
                        "true_index": true_index,
                        "pred_class": CLASS_NAMES[prediction],
                        "pred_index": prediction,
                        "correct": prediction == true_index,
                    }
                )

    return pd.DataFrame(rows)


def _select_diverse_rows(
    candidates: pd.DataFrame,
    count: int,
    class_order: Iterable[str],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    used_paths: set[str] = set()

    for class_name in class_order:
        if len(selected) >= count:
            break
        class_rows = candidates[candidates["true_class"] == class_name]
        if class_rows.empty:
            continue
        row = class_rows.sort_values(["image_path"]).iloc[0]
        selected.append(row.to_dict())
        used_paths.add(row["image_path"])

    if len(selected) < count:
        remaining = candidates[~candidates["image_path"].isin(used_paths)]
        for _, row in remaining.sort_values(["true_class", "image_path"]).iterrows():
            if len(selected) >= count:
                break
            selected.append(row.to_dict())

    return selected


def select_attention_examples(
    predictions: pd.DataFrame,
    condition_labels: Iterable[str],
    correct_per_condition: int = 3,
    error_per_condition: int = 1,
    class_order: Iterable[str] = CLASS_NAMES,
) -> pd.DataFrame:
    selected_rows: list[dict[str, object]] = []

    for condition_label in condition_labels:
        condition_rows = predictions[predictions["condition"] == condition_label]
        correct_rows = condition_rows[condition_rows["correct"]]
        error_rows = condition_rows[~condition_rows["correct"]]

        for row in _select_diverse_rows(
            correct_rows,
            correct_per_condition,
            class_order,
        ):
            row["case_type"] = "correct"
            row["source_condition"] = condition_label
            selected_rows.append(row)

        for row in _select_diverse_rows(
            error_rows,
            error_per_condition,
            class_order,
        ):
            row["case_type"] = "error"
            row["source_condition"] = condition_label
            selected_rows.append(row)

    return pd.DataFrame(selected_rows).reset_index(drop=True)
