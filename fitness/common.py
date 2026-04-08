from __future__ import annotations

import numpy as np
from PIL import Image

VISIBLE_BACKGROUND = (255, 255, 255, 255)


def image_to_visible_rgb_u8(image: Image.Image) -> np.ndarray:
    base = Image.new("RGBA", image.size, VISIBLE_BACKGROUND)
    visible = Image.alpha_composite(base, image.convert("RGBA")).convert("RGB")
    return np.ascontiguousarray(np.asarray(visible, dtype=np.uint8))


def mean_absolute_error_rgb_u8(source_array: np.ndarray, candidate_array: np.ndarray) -> float:
    diff = np.subtract(
        source_array.astype(np.uint8, copy=False),
        candidate_array.astype(np.uint8, copy=False),
        dtype=np.int16,
    )
    return float(np.mean(np.abs(diff), dtype=np.float32))


def mean_squared_error_rgb_u8(source_array: np.ndarray, candidate_array: np.ndarray) -> float:
    diff = np.subtract(
        source_array.astype(np.uint8, copy=False),
        candidate_array.astype(np.uint8, copy=False),
        dtype=np.int16,
    )
    return float(np.mean(np.square(diff, dtype=np.int32), dtype=np.float32))
