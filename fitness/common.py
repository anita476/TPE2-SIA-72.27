from __future__ import annotations

import numpy as np
from PIL import Image


def candidate_to_rgba_f64(candidate: Image.Image) -> np.ndarray:
    return np.asarray(candidate.convert("RGBA"), dtype=np.float64)


def source_as_f64(source_array: np.ndarray) -> np.ndarray:
    return source_array.astype(np.float64, copy=False)
