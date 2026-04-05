from __future__ import annotations

import math
import random

from utils.genetic import Individual


def elite_selection(
    population: list[Individual],
    fitness_scores: list[float],
    k: int,
    rng: random.Random,
) -> list[Individual]:
    """Select K individuals by ranking them and repeating each one n(i) times."""
    n = len(population)
    paired = sorted(zip(fitness_scores, population), key=lambda x: x[0])
    selected = []
    for i, (_, individual) in enumerate(paired):
        repeats = math.ceil((k - i) / n)
        if repeats > 0:
            selected.extend([individual] * repeats)

    return selected[:k]
