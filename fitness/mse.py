from __future__ import annotations

import numpy as np
from PIL import Image

from .common import candidate_to_rgba_f64, source_as_f64


def mse_fitness(source_array: np.ndarray, candidate: Image.Image) -> float:
    """Mean squared error fitness over RGBA; higher is better (1 = perfect match)."""
    diff = source_as_f64(source_array) - candidate_to_rgba_f64(candidate)
    mse = float(np.mean(diff * diff))
    return 1.0 - mse / 65025.0
