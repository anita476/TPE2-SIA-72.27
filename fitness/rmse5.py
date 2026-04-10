from __future__ import annotations

import numpy as np

from .common import mean_squared_error_rgb_u8


def rmse5_fitness(source_array: np.ndarray, candidate_array: np.ndarray) -> float:
    """RMSE fitness compressed with power 5; higher is better (1 = perfect match)."""
    rmse = np.sqrt(mean_squared_error_rgb_u8(source_array, candidate_array))
    base_fitness = 1.0 - rmse / 255.0
    return float(base_fitness**5)
