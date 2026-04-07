import random
from utils.genetic import Individual

def tournament_stochastic(
        population: list[Individual],
        fitness_scores: list[float],
        k: int,
        rng: random.Random,
) -> list[Individual]:
    """
    Se elige un Threshold de [0.5,1].
    De la población original de tamaño N, se eligen solo 2 individuos al azar. Se toma un valor r al azar uniformemente entre r ← U [0, 1).
    Si r < T hreshold se selecciona el más apto.
    Caso contrario, se selecciona el menos apto.
    Se repite el proceso hasta conseguir los K individuos que se precisan.
    """
    subset_num = 2
    selected = []
    threshold = 0.5 # @todo

    for _ in range(k):
        contenders_idx = rng.sample(range(len(population)), subset_num)

        # reverse is fall bc we need creciente
        fittest_idx, weakest_idx = sorted(
            contenders_idx, key=lambda idx: fitness_scores[idx], reverse=False
        )
        r = rng.random()  # U[0, 1)
        winner_idx = fittest_idx if r < threshold else weakest_idx

        selected.append(population[winner_idx])

    return selected