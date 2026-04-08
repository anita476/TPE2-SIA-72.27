from __future__ import annotations

import math

import numpy as np

from .common import mean_squared_error_rgb_u8


def rmse_fitness(source_array: np.ndarray, candidate_array: np.ndarray) -> float:
    """RMSE-based fitness over visible RGB; higher is better (1 = perfect match)."""
    rmse = math.sqrt(mean_squared_error_rgb_u8(source_array, candidate_array))
    return 1.0 - rmse / 255.0
