from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

# (generation, best_score, all_fitness_scores) -> True means stop
StopCondition = Callable[[int, float, list[float]], bool]

def target_fitness(threshold: float) -> StopCondition:
    """Stop when the best fitness reaches or goes below the threshold."""
    def condition(_generation: int, best_score: float, _fitness_scores: list[float]) -> bool:
        return best_score >= threshold
    return condition

"""
def no_improvement(window: int, min_delta: float = 1e-4) -> StopCondition:
    history: list[float] = []

    def condition(_generation: int, best_score: float, _fitness_scores: list[float]) -> bool:
        history.append(best_score)
        if len(history) < window:
            return False
        return (history[-1] - history[-window]) < min_delta

    return condition
"""
def no_improvement(window: int, min_delta: float = 1e-4) -> StopCondition:
    history: list[float] = []
    def condition(_generation: int, best_score: float, _fitness_scores: list[float]) -> bool:
        history.append(best_score)
        if len(history) < window:
            return False
        # max improvement over entire window, not just endpoints
        return (max(history[-window:]) - min(history[-window:])) < min_delta
    return condition

def population_converged(min_std: float = 1e-3) -> StopCondition:
    """Stop when the std dev of fitness scores drops below min_std (population has converged)."""
    def condition(_generation: int, _best_score: float, fitness_scores: list[float]) -> bool:
        return float(np.std(fitness_scores)) < min_std
    return condition


def time_limit(seconds: float) -> StopCondition:
    """Stop after a wall-clock time limit."""
    start = time.monotonic()

    def condition(_generation: int, _best_score: float, _fitness_scores: list[float]) -> bool:
        return (time.monotonic() - start) >= seconds

    return condition


def any_of(*conditions: StopCondition) -> StopCondition:
    """Stop when any of the given conditions is met."""
    def condition(generation: int, best_score: float, fitness_scores: list[float]) -> bool:
        return any(c(generation, best_score, fitness_scores) for c in conditions)
    return condition
