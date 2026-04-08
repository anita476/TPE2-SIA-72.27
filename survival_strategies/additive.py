from __future__ import annotations

import random

from utils.genetic import Individual

from .common import PopulationEvaluator, Selector


def additive_survival(
    population: list[Individual],
    population_fitness: list[float],
    offspring: list[Individual],
    evaluator: PopulationEvaluator,
    selector: Selector,
    population_size: int,
    rng: random.Random,
) -> list[Individual]:
    """Select the next generation from the current population plus all offspring."""
    offspring_fitness = evaluator.evaluate_population(offspring)
    candidates = list(population) + list(offspring)
    candidate_fitness = population_fitness + offspring_fitness
    return selector(candidates, candidate_fitness, population_size, rng)
