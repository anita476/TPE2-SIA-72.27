from __future__ import annotations

import random

import numpy as np

from utils.genetic import Individual

from .common import FitnessFn, Selector, evaluate_population_fitness


def additive_survival(
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
    """Select the next generation from the current population plus all offspring."""
    offspring_fitness = evaluate_population_fitness(offspring, source_array, image_size, fitness_fn)
    candidates = list(population) + list(offspring)
    candidate_fitness = population_fitness + offspring_fitness
    return selector(candidates, candidate_fitness, population_size, rng)
