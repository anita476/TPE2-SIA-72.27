from __future__ import annotations

import random

from genetic_utils import Individual

_EPSILON = 1e-9


def roulette_selection(
    population: list[Individual],
    fitness_scores: list[float],
    k: int,
    rng: random.Random,
) -> list[Individual]:
    """Select K individuals using fitness-proportionate (roulette wheel) selection.

    Lower MAE translates to higher selection probability.
    Each of the K selections draws an independent uniform random number.
    """
    weights = [1.0 / (f + _EPSILON) for f in fitness_scores]
    total = sum(weights)
    cumulative = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cumulative.append(acc)

    selected = []
    for _ in range(k):
        r = rng.random()
        for i, q in enumerate(cumulative):
            if r <= q:
                selected.append(population[i])
                break
        else:
            selected.append(population[-1])

    return selected
