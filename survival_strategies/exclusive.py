from __future__ import annotations

import random

from utils.genetic import Individual

from .common import PopulationEvaluator, Selector


def exclusive_survival(
    population: list[Individual],
    population_fitness: list[float],
    offspring: list[Individual],
    evaluator: PopulationEvaluator,
    selector: Selector,
    population_size: int,
    rng: random.Random,
) -> list[Individual]:
    """Build the next generation from offspring first, then selected current individuals if needed."""
    offspring_count = len(offspring)

    if offspring_count > population_size:
        offspring_fitness = evaluator.evaluate_population(offspring)
        return selector(offspring, offspring_fitness, population_size, rng)

    if offspring_count == population_size:
        return list(offspring)

    survivors_from_population = selector(
        population,
        population_fitness,
        population_size - offspring_count,
        rng,
    )
    return list(offspring) + survivors_from_population
