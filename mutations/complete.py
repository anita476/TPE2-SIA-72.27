from __future__ import annotations

import random

from utils.genetic import Individual, mutate_gene


def complete_mutation(
    individual: Individual,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    mutation_rate: float,
    mutation_strength: float,
) -> Individual:
    """With probability mutation_rate, mutate every gene in the individual."""
    if rng.random() >= mutation_rate:
        return individual

    return [mutate_gene(gene, bounds, rng, mutation_strength) for gene in individual]
