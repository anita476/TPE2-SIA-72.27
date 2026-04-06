from __future__ import annotations

import random

from utils.genetic import Individual, mutate_gene


def gen_mutation(
    individual: Individual,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    mutation_rate: float,
    mutation_strength: float,
) -> Individual:
    """With probability mutation_rate, mutate exactly one randomly chosen gene."""
    if rng.random() >= mutation_rate:
        return individual

    idx = rng.randrange(len(individual))
    return [
        mutate_gene(gene, bounds, rng, mutation_strength) if i == idx else gene
        for i, gene in enumerate(individual)
    ]
