from __future__ import annotations

import random

from utils.genetic import Individual, mutate_gene


def multigen_uniform_mutation(
    individual: Individual,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    mutation_rate: float,
    mutation_strength: float,
) -> Individual:
    """Each gene is independently mutated with probability mutation_rate."""
    return [
        mutate_gene(gene, bounds, rng, mutation_strength)
        if rng.random() < mutation_rate
        else gene
        for gene in individual
    ]
