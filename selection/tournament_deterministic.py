import random
from utils.genetic import Individual

def tournament_deterministic(
    population: list[Individual],
    fitness_scores: list[float],
    k: int,
    rng: random.Random,
) -> list[Individual]:
    """
    De la población original de tamaño N, se eligen M individuos al azar.
    De los M individuos, se elige el mejor.
    Se repite el proceso hasta conseguir los K individuos que se precisan.
    """
    subset_num = 5 # @todo change later to be configurable
    selected = []

    for _ in range(k):
        contenders_idx = rng.sample(range(len(population)), subset_num)

        #  index with the best  fitness score is chosen -> now its higher = better
        winner_idx = max(contenders_idx, key=lambda idx: fitness_scores[idx])

        selected.append(population[winner_idx])

    return selected