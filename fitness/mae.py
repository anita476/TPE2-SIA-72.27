from __future__ import annotations

import numpy as np
from PIL import Image

from .common import candidate_to_rgba_f64, source_as_f64


def mae_fitness(source_array: np.ndarray, candidate: Image.Image) -> float:
    """Mean absolute error over RGBA channels; lower is better."""
    diff = source_as_f64(source_array) - candidate_to_rgba_f64(candidate)
    return float(np.mean(np.abs(diff)))
