from __future__ import annotations

import numpy as np

from .common import mean_squared_error_rgb_u8


def mse5_fitness(source_array: np.ndarray, candidate_array: np.ndarray) -> float:
    """MSE fitness compressed with power 5; higher is better (1 = perfect match)."""
    mse = mean_squared_error_rgb_u8(source_array, candidate_array)
    base_fitness = 1.0 - mse / 65025.0
    return base_fitness**5
