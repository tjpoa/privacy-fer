from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

LEFT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
LEFT_EYEBROW = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
RIGHT_EYEBROW = (336, 296, 334, 293, 300, 285, 295, 282, 283, 276)
NOSE = (1, 2, 98, 327, 168, 197, 195, 5, 4, 45, 275, 220, 440, 115, 344)
MOUTH = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
)

REGION_ORDER = (
    "eyes",
    "eyebrows",
    "nose",
    "mouth",
    "face_other",
    "outside_face",
)


@dataclass(frozen=True)
class LandmarkAttentionResult:
    landmarks_detected: bool
    num_landmarks: int
    metrics: dict[str, float]


def is_mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return True


def _import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for landmark-guided attention analysis. "
            "Install it with: python -m pip install mediapipe"
        ) from exc
    return mp


@contextmanager
def create_face_landmarker(
    model_path: str | Path,
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
) -> Iterator[object]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Face Landmarker model not found: {model_path}. "
            "Download face_landmarker.task and place it there."
        )

    mp = _import_mediapipe()
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        yield landmarker


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        rgb = image
    else:
        raise ValueError("image must be grayscale or RGB with 3 channels")

    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)

    rgb = rgb.astype(np.float32)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))


def detect_face_landmarks(
    landmarker: object,
    image: np.ndarray,
    min_detection_size: int = 256,
) -> np.ndarray | None:
    mp = _import_mediapipe()
    rgb = _to_uint8_rgb(image)
    height, width = rgb.shape[:2]
    scale = max(1.0, float(min_detection_size) / float(min(height, width)))

    if scale > 1.0:
        detection_width = int(round(width * scale))
        detection_height = int(round(height * scale))
        detection_rgb = cv2.resize(
            rgb,
            (detection_width, detection_height),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        detection_rgb = rgb
        detection_height, detection_width = height, width

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detection_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    points = [
        (
            float(landmark.x) * detection_width / scale,
            float(landmark.y) * detection_height / scale,
        )
        for landmark in result.face_landmarks[0]
    ]
    return np.asarray(points, dtype=np.float32)


def _convex_hull_mask(
    points: np.ndarray,
    image_shape: tuple[int, int],
    indices: tuple[int, ...],
    dilation: int = 0,
) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    selected_points = []
    for index in indices:
        if index >= len(points):
            continue
        x_coord, y_coord = points[index]
        if np.isfinite(x_coord) and np.isfinite(y_coord):
            selected_points.append([x_coord, y_coord])

    if len(selected_points) < 3:
        return mask.astype(bool)

    selected = np.asarray(selected_points, dtype=np.float32)
    selected[:, 0] = np.clip(selected[:, 0], 0, width - 1)
    selected[:, 1] = np.clip(selected[:, 1], 0, height - 1)
    hull = cv2.convexHull(np.rint(selected).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)

    if dilation > 0:
        kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask.astype(bool)


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(bool)

    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _combined_hull_mask(
    points: np.ndarray,
    image_shape: tuple[int, int],
    index_groups: tuple[tuple[int, ...], ...],
    dilation: int = 0,
) -> np.ndarray:
    combined = np.zeros(image_shape, dtype=bool)
    for indices in index_groups:
        combined |= _convex_hull_mask(
            points=points,
            image_shape=image_shape,
            indices=indices,
            dilation=dilation,
        )
    return combined


def build_face_region_masks(
    landmark_points: np.ndarray,
    image_shape: tuple[int, int],
    target_shape: tuple[int, int] | None = None,
    dilation_ratio: float = 0.025,
) -> dict[str, np.ndarray]:
    height, width = image_shape
    dilation = max(1, int(round(min(height, width) * dilation_ratio)))

    face_mask = _convex_hull_mask(
        landmark_points,
        image_shape,
        FACE_OVAL,
        dilation=dilation * 2,
    )
    raw_masks = {
        "eyes": _combined_hull_mask(
            landmark_points,
            image_shape,
            (LEFT_EYE, RIGHT_EYE),
            dilation=dilation,
        ),
        "eyebrows": _combined_hull_mask(
            landmark_points,
            image_shape,
            (LEFT_EYEBROW, RIGHT_EYEBROW),
            dilation=dilation,
        ),
        "nose": _convex_hull_mask(
            landmark_points,
            image_shape,
            NOSE,
            dilation=dilation,
        ),
        "mouth": _convex_hull_mask(
            landmark_points,
            image_shape,
            MOUTH,
            dilation=dilation,
        ),
    }

    exclusive_masks: dict[str, np.ndarray] = {}
    reserved = np.zeros((height, width), dtype=bool)
    for region_name in ("eyes", "eyebrows", "nose", "mouth"):
        region_mask = raw_masks[region_name] & face_mask & ~reserved
        exclusive_masks[region_name] = region_mask
        reserved |= region_mask

    exclusive_masks["face_other"] = face_mask & ~reserved
    exclusive_masks["outside_face"] = ~face_mask

    if target_shape is not None:
        exclusive_masks = {
            region_name: _resize_mask(mask, target_shape)
            for region_name, mask in exclusive_masks.items()
        }

    return exclusive_masks


def attention_region_metrics(
    attention_map: np.ndarray,
    masks: dict[str, np.ndarray],
    eps: float = 1e-12,
) -> dict[str, float]:
    attention = np.asarray(attention_map, dtype=np.float64)
    attention = np.clip(attention, 0.0, None)
    total_attention = float(attention.sum())

    metrics: dict[str, float] = {}
    for region_name in REGION_ORDER:
        mask = masks[region_name]
        if mask.shape != attention.shape:
            mask = _resize_mask(mask, attention.shape)

        if total_attention <= eps:
            mass = 0.0
        else:
            mass = float(attention[mask].sum() / total_attention)

        metrics[f"{region_name}_attention_mass"] = mass
        metrics[f"{region_name}_area_fraction"] = float(mask.mean())

    return metrics


def compute_landmark_attention_metrics(
    landmarker: object,
    image: np.ndarray,
    attention_map: np.ndarray,
) -> LandmarkAttentionResult:
    landmark_points = detect_face_landmarks(landmarker, image)
    if landmark_points is None:
        empty_metrics = {
            f"{region_name}_{suffix}": 0.0
            for region_name in REGION_ORDER
            for suffix in ("attention_mass", "area_fraction")
        }
        return LandmarkAttentionResult(
            landmarks_detected=False,
            num_landmarks=0,
            metrics=empty_metrics,
        )

    masks = build_face_region_masks(
        landmark_points=landmark_points,
        image_shape=image.shape[:2],
        target_shape=attention_map.shape[:2],
    )
    return LandmarkAttentionResult(
        landmarks_detected=True,
        num_landmarks=int(len(landmark_points)),
        metrics=attention_region_metrics(attention_map, masks),
    )


def draw_region_overlay(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    alpha: float = 0.35,
) -> np.ndarray:
    rgb = _to_uint8_rgb(image)
    colors = {
        "eyes": (46, 134, 171),
        "eyebrows": (242, 149, 89),
        "nose": (102, 187, 106),
        "mouth": (231, 76, 60),
        "face_other": (160, 160, 160),
        "outside_face": (0, 0, 0),
    }

    overlay = rgb.copy()
    for region_name in REGION_ORDER:
        if region_name == "outside_face":
            continue
        mask = _resize_mask(masks[region_name], rgb.shape[:2])
        color = np.asarray(colors[region_name], dtype=np.uint8)
        overlay[mask] = color

    return cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)
