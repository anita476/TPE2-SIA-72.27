from __future__ import annotations

import numpy as np
from PIL import Image

from .common import candidate_to_rgba_f64, source_as_f64


def mse_fitness(source_array: np.ndarray, candidate: Image.Image) -> float:
    """Mean squared error over RGBA; lower is better."""
    diff = source_as_f64(source_array) - candidate_to_rgba_f64(candidate)
    return float(np.mean(diff * diff))
