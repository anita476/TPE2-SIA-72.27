from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .mse import mse_fitness


def rmse_fitness(source_array: np.ndarray, candidate: Image.Image) -> float:
    """Root mean squared error (RMSE); lower is better. Ranks individuals the same as MSE."""
    return math.sqrt(mse_fitness(source_array, candidate))
