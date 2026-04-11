from __future__ import annotations

import random

from utils.genetic import Individual


def one_point_crossover(
    parent1: Individual,
    parent2: Individual,
    rng: random.Random,
) -> tuple[Individual, Individual]:
    """Perform one-point crossover between two parent individuals to produce two child individuals."""
    length = len(parent1)

    pt = rng.randint(0, length - 1)
    child1 = parent1[:pt] + parent2[pt:]
    child2 = parent2[:pt] + parent1[pt:]
    return child1, child2
