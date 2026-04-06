
import bisect
import itertools
import random

from utils.genetic import Individual

def ranking(
    population: list[Individual],
    fitness_scores: list[float],
    k: int,
    rng: random.Random,
) -> list[Individual]:
    """
    f'(i) = (N - rank(i)) / N
    """
    n = len(population)

    # lower fitness score = better = lower rank index
    sorted_indices = sorted(range(n), key=lambda i: fitness_scores[i])

    ranks = [0] * n
    for j, ind in enumerate(sorted_indices):
        ranks[ind] = j
    weights = [(n - ranks[i]) / n for i in range(n)]

    # normalize to sum 1
    total = sum(weights)
    probabilities = [w / total for w in weights]

    # build cumulative distribution
    cumulative = list(itertools.accumulate(probabilities))

    selected = []
    for _ in range(k):
        selected.append(population[bisect.bisect_left(cumulative, rng.random())])

    return selected
