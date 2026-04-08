from __future__ import annotations

import numpy as np

from .common import mean_absolute_error_rgb_u8


def mae_fitness(source_array: np.ndarray, candidate_array: np.ndarray) -> float:
    """Mean absolute error fitness over visible RGB; higher is better (1 = perfect match)."""
    mae = mean_absolute_error_rgb_u8(source_array, candidate_array)
    return 1.0 - mae / 255.0
