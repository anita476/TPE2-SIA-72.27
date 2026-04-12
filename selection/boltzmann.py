import bisect
import itertools
import math
import random
from collections.abc import Callable

from utils.genetic import Individual

TemperatureSchedule = Callable[[int], float]


def boltzmann(
    population: list[Individual],
    fitness_scores: list[float],
    k: int,
    rng: random.Random,
    temperature: float,
) -> list[Individual]:
    """
    f_norm(i) = f(i) / mean(f)

    When the population is diverse: outliers get amplified → strong selection pressure.
    When the population has converged: all f_norm ≈ 1 → selection becomes near-uniform,
    naturally preventing lock-in on a local optimum.

    At high T: near-uniform selection (exploration).
    At low T: strongly favors individuals above the mean (exploitation).

    boltzmann_factor(i) = exp(f_norm(i) / T)
    p(i) = boltzmann_factor(i) / sum(boltzmann_factor)
    """
    avg_f = sum(fitness_scores) / len(fitness_scores)

    if avg_f < 1e-9:
        return [population[i] for i in (rng.randrange(len(population)) for _ in range(k))]

    normalized = [f / avg_f for f in fitness_scores]
    boltzmann_factors = [math.exp(n / temperature) for n in normalized]
    total = sum(boltzmann_factors)

    cumulative = list(itertools.accumulate(v / total for v in boltzmann_factors))

    selected = []
    for _ in range(k):
        selected.append(population[bisect.bisect_left(cumulative, rng.random())])

    return selected


class AnnealedBoltzmann:
    """
    Wraps boltzmann() with the schedule (from the presentation: T(t) = T_c + (T_0 - T_c) * e^(k*t)
    Fits the standard Selector signature so run_genetic_algorithm needs no changes.
    """

    def __init__(self, t0: float, t_c: float, k: float) -> None:
        self.t0 = t0
        self.t_c = t_c
        self.k = k
        self.generation = 0

    def __call__(
        self,
        population: list[Individual],
        fitness_scores: list[float],
        k: int,
        rng: random.Random,
    ) -> list[Individual]:
        temperature = self.t_c + (self.t0 - self.t_c) * math.exp(self.k * self.generation)
        self.generation += 1
        return boltzmann(population, fitness_scores, k, rng, temperature)
