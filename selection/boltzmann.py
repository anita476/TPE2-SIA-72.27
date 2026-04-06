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
    boltzmann_factor(i) = e^(f(i) / T)
    expval(i)           = boltzmann_factor(i) / <boltzmann_factor(x)>_g
    Selection probability is proportional to how much better than the population average the individual is.
    Higher temperature flattens differences -> more exploration; lower temperature sharpens them -> more exploitation.
    """
    boltzmann_factors = [math.exp(fitness_scores[i] / temperature) for i in range(len(population))]
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
