import random
from utils.genetic import Individual

def tournament_deterministic(
    population: list[Individual],
    fitness_scores: list[float],
    k: int,
    rng: random.Random,
    tournament_size: int = 2,
) -> list[Individual]:
    """
    De la población original de tamaño N, se eligen M individuos al azar (tournament_size).
    De los M individuos, se elige el mejor.
    Se repite el proceso hasta conseguir los K individuos que se precisan.
    
    Args:
        population: Current population
        fitness_scores: Fitness values for each individual
        k: Number of individuals to select
        rng: Random number generator
        tournament_size: Number of contenders in each tournament (M parameter)
    """
    selected = []

    for _ in range(k):
        contenders_idx = rng.sample(range(len(population)), tournament_size)

        #  index with the best  fitness score is chosen -> now its higher = better
        winner_idx = max(contenders_idx, key=lambda idx: fitness_scores[idx])

        selected.append(population[winner_idx])

    return selected