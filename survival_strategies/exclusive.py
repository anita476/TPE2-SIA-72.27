from __future__ import annotations

import random

import numpy as np

from utils.genetic import Individual

from .common import FitnessFn, Selector, evaluate_population_fitness


def exclusive_survival(
    population: list[Individual],
    population_fitness: list[float],
    offspring: list[Individual],
    source_array: np.ndarray,
    image_size: tuple[int, int],
    fitness_fn: FitnessFn,
    selector: Selector,
    population_size: int,
    rng: random.Random,
) -> list[Individual]:
    """Build the next generation from offspring first, then selected current individuals if needed."""
    offspring_count = len(offspring)

    if offspring_count > population_size:
        offspring_fitness = evaluate_population_fitness(offspring, source_array, image_size, fitness_fn)
        return selector(offspring, offspring_fitness, population_size, rng)

    if offspring_count == population_size:
        return list(offspring)

    survivors_from_population = selector(
        population,
        population_fitness,
        population_size - offspring_count,
        rng,
    )
    return list(offspring) + survivors_from_population
