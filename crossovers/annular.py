from __future__ import annotations

from math import ceil
import random

from utils.genetic import Individual


def annular_crossover(
    parent1: Individual,
    parent2: Individual,
    rng: random.Random,
) -> tuple[Individual, Individual]:
    """Swap a circular segment starting at a random locus."""
    length_parent1 = len(parent1)
    length_parent2 = len(parent2)

    if length_parent1 == 0 or length_parent1 != length_parent2:
        return list(parent1), list(parent2)

    start = rng.randint(0, length_parent1 - 1)
    segment_length = rng.randint(0, ceil(length_parent1 / 2))

    child1 = list(parent1)
    child2 = list(parent2)

    for offset in range(segment_length):
        index = (start + offset) % length_parent1
        child1[index], child2[index] = child2[index], child1[index]

    return child1, child2
