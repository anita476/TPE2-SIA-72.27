from __future__ import annotations

import numpy as np
from PIL import Image

from .common import candidate_to_rgba_f64, source_as_f64


def mae_fitness(source_array: np.ndarray, candidate: Image.Image) -> float:
    """Mean absolute error fitness over RGBA channels; higher is better (1 = perfect match)."""
    diff = source_as_f64(source_array) - candidate_to_rgba_f64(candidate)
    mae = float(np.mean(np.abs(diff)))
    return 1.0 - mae / 255.0
