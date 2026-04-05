from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
from PIL import Image

from utils.genetic import Individual
from utils.image import create_phenotype_image

Selector = Callable[[list[Individual], list[float], int, random.Random], list[Individual]]
FitnessFn = Callable[[np.ndarray, Image.Image], float]
SurvivalStrategy = Callable[
    [
        list[Individual],
        list[float],
        list[Individual],
        np.ndarray,
        tuple[int, int],
        FitnessFn,
        Selector,
        int,
        random.Random,
    ],
    list[Individual],
]


def evaluate_population_fitness(
    population: list[Individual],
    source_array: np.ndarray,
    image_size: tuple[int, int],
    fitness_fn: FitnessFn,
) -> list[float]:
    """Compute fitness for each individual."""
    return [
        fitness_fn(source_array, create_phenotype_image(individual, image_size=image_size))
        for individual in population
    ]
