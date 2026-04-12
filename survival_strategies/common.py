from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeAlias

import numpy as np
from PIL import Image

from fitness.common import image_to_visible_rgb_u8
from utils.genetic import Gene, Individual
from utils.image import render_individual_visible_array

_WORKERS = 8
_pool = ThreadPoolExecutor(max_workers=_WORKERS)

Selector = Callable[[list[Individual], list[float], int, random.Random], list[Individual]]
FitnessFn = Callable[[np.ndarray, np.ndarray], float]
IndividualKey: TypeAlias = tuple[Gene, ...]


class PopulationEvaluator:
    def __init__(
        self,
        source_image: Image.Image,
        image_size: tuple[int, int],
        fitness_fn: FitnessFn,
    ) -> None:
        self.source_array = image_to_visible_rgb_u8(source_image)
        self.image_size = image_size
        self.fitness_fn = fitness_fn
        self._fitness_cache: dict[IndividualKey, float] = {}

    def render(self, individual: Individual) -> np.ndarray:
        return render_individual_visible_array(individual, image_size=self.image_size)

    def _compute_fitness(self, individual: Individual) -> float:
        return self.fitness_fn(self.source_array, self.render(individual))

    def evaluate_population(self, population: list[Individual]) -> list[float]:
        scores = [0.0] * len(population)
        pending: dict[IndividualKey, tuple[Individual, list[int]]] = {}

        for index, individual in enumerate(population):
            key = tuple(individual)
            cached = self._fitness_cache.get(key)
            if cached is not None:
                scores[index] = cached
                continue

            entry = pending.get(key)
            if entry is None:
                pending[key] = (individual, [index])
            else:
                entry[1].append(index)

        if not pending:
            return scores

        pending_items = list(pending.items())
        pending_individuals = [individual for _, (individual, _) in pending_items]

        for (key, (_, indices)), score in zip(pending_items, _pool.map(self._compute_fitness, pending_individuals)):
            self._fitness_cache[key] = score # @todo cache is never evicted, accumulates endlessly (probs fine for our nb of gens)
            for index in indices:
                scores[index] = score

        return scores


SurvivalStrategy = Callable[
    [
        list[Individual],
        list[float],
        list[Individual],
        PopulationEvaluator,
        Selector,
        int,
        random.Random,
    ],
    list[Individual],
]
