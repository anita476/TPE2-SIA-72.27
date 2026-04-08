from __future__ import annotations

import numpy as np

from .common import mean_squared_error_rgb_u8


def mse_fitness(source_array: np.ndarray, candidate_array: np.ndarray) -> float:
    """Mean squared error fitness over visible RGB; higher is better (1 = perfect match)."""
    mse = mean_squared_error_rgb_u8(source_array, candidate_array)
    return 1.0 - mse / 65025.0
