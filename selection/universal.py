from __future__ import annotations

import random

from utils.genetic import Individual

_EPSILON = 1e-9


def universal_selection(
    population: list[Individual],
    fitness_scores: list[float],
    k: int,
    rng: random.Random,
) -> list[Individual]:
    """Select K individuals using Stochastic Universal Sampling (SUS).

    Same relative weights as roulette but uses a single random offset r and
    K equally-spaced pointers r_j = (r + j) / K to traverse the CDF once,
    reducing selection variance compared to independent roulette spins.
    """
    weights = [max(f, 0.0) for f in fitness_scores]
    total = sum(weights)
    if total <= _EPSILON:
        weights = [1.0 for _ in fitness_scores]
        total = float(len(weights))

    cumulative = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cumulative.append(acc)

    r = rng.random()
    pointers = [(r + j) / k for j in range(k)]

    selected = []
    idx = 0
    for pointer in pointers:
        while idx < len(cumulative) - 1 and cumulative[idx] < pointer:
            idx += 1
        selected.append(population[idx])

    return selected
