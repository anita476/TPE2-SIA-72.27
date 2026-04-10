from __future__ import annotations

import random

from utils.genetic import Individual


UNIFORM_CROSSOVER_PROBABILITY = 0.5


def uniform_crossover(
    parent1: Individual,
    parent2: Individual,
    rng: random.Random,
) -> tuple[Individual, Individual]:
    """Swap each gene independently with fixed probability."""
    child1 = list(parent1)
    child2 = list(parent2)

    for index in range(len(parent1)):
        if rng.random() < UNIFORM_CROSSOVER_PROBABILITY:
            child1[index], child2[index] = child2[index], child1[index]

    return child1, child2
