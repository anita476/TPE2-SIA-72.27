from __future__ import annotations

import random

from utils.genetic import Individual, mutate_gene


def multigen_limited_mutation(
    individual: Individual,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    mutation_rate: float,
    mutation_strength: float,
) -> Individual:
    """With probability mutation_rate, mutate a random number [1, M] of genes."""
    if rng.random() >= mutation_rate:
        return individual

    m = len(individual)
    count = rng.randint(1, m)
    indices_to_mutate = set(rng.sample(range(m), count))
    return [
        mutate_gene(gene, bounds, rng, mutation_strength) if i in indices_to_mutate else gene
        for i, gene in enumerate(individual)
    ]
