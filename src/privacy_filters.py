from __future__ import annotations

import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image cannot be None")
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image cannot be empty")
    if image.ndim not in (2, 3):
        raise ValueError("image must be 2D (grayscale) or 3D (color)")
    return image


def _to_unit_range(image: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    image = _validate_image(image)
    original_dtype = image.dtype

    if np.issubdtype(original_dtype, np.integer):
        max_value = float(np.iinfo(original_dtype).max)
        return image.astype(np.float32) / max_value, original_dtype

    image = image.astype(np.float32)
    if image.min() >= 0.0 and image.max() <= 1.0:
        return image, original_dtype

    return np.clip(image / 255.0, 0.0, 1.0), original_dtype


def _restore_dtype(image: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)

    if np.issubdtype(original_dtype, np.integer):
        max_value = float(np.iinfo(original_dtype).max)
        return np.rint(image * max_value).astype(original_dtype)

    return image.astype(original_dtype)


def apply_gaussian_blur(image: np.ndarray, level: float) -> np.ndarray:
    image = _validate_image(image)

    if level < 0:
        raise ValueError("level must be non-negative")
    if level == 0:
        return image.copy()

    kernel_size = max(3, 2 * int(np.ceil(level)) + 1)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=float(level))


def apply_canny_edges(image: np.ndarray) -> np.ndarray:
    image = _validate_image(image)

    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image

    if grayscale.dtype != np.uint8:
        grayscale = np.clip(grayscale, 0, 255).astype(np.uint8)

    median_intensity = np.median(grayscale)
    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))

    if upper <= lower:
        upper = min(255, lower + 50)

    return cv2.Canny(grayscale, threshold1=lower, threshold2=upper)


def apply_diffusion_noise(image: np.ndarray, t_step: float) -> np.ndarray:
    image = _validate_image(image)

    if t_step < 0:
        raise ValueError("t_step must be non-negative")

    noise_strength = float(t_step)
    if noise_strength > 1.0:
        noise_strength = min(noise_strength / 1000.0, 1.0)

    image_unit, original_dtype = _to_unit_range(image)
    noise = np.random.normal(loc=0.0, scale=1.0, size=image.shape).astype(np.float32)

    noisy_image = (
        np.sqrt(1.0 - noise_strength) * image_unit
        + np.sqrt(noise_strength) * noise
    )
    noisy_image = np.clip(noisy_image, 0.0, 1.0)

    return _restore_dtype(noisy_image, original_dtype)
