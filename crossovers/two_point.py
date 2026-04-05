from __future__ import annotations

import random

from utils.genetic import Individual

from .one_point import one_point_crossover


def two_point_crossover(
    parent1: Individual,
    parent2: Individual,
    rng: random.Random,
) -> tuple[Individual, Individual]:
    """Perform two-point crossover between two parent individuals to produce two child individuals."""
    length = len(parent1)

    if length < 3:
        return one_point_crossover(parent1, parent2, rng)

    pt1, pt2 = sorted(rng.sample(range(1, length), 2))
    child1 = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
    child2 = parent2[:pt1] + parent1[pt1:pt2] + parent2[pt2:]
    return child1, child2
