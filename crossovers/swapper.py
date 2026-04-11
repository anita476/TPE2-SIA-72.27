from __future__ import annotations

import random

from crossovers.uniform import uniform_crossover
from utils.genetic import Individual

PROB_SWAP = 0.2

def swapper_crossover(
    parent1: Individual,
    parent2: Individual,
    rng: random.Random,
) -> tuple[Individual, Individual]:
    """Perform swapper crossover between two parent individuals to produce two child individuals.
    With a fixed probability, perform uniform crossover. Otherwise, randomly shuffle the combined genes of both parents and split them into two children."""
    if rng.random() >= PROB_SWAP:
        return uniform_crossover(parent1, parent2, rng)

    length = len(parent1)
    if length == 0 or length != len(parent2):
        return list(parent1), list(parent2)

    pool = list(parent1) + list(parent2)
    rng.shuffle(pool)
    child1 = pool[:length]
    child2 = pool[length:]
    return child1, child2
