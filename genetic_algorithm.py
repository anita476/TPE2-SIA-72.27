from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from utils.stop_conditions import StopCondition
from survival_strategies.common import SurvivalStrategy, evaluate_population_fitness
from utils.genetic import Individual, random_triangle, get_overflow_bounds
from utils.image import create_phenotype_image, save_phenotype_image


@dataclass
class GAResult:
    image_bytes: bytes
    best_fitness: float
    generations_run: int


Selector = Callable[[list[Individual], list[float], int, random.Random], list[Individual]]
Crossover = Callable[[Individual, Individual, random.Random], tuple[Individual, Individual]]
FitnessFn = Callable[[np.ndarray, Image.Image], float]
MutationFn = Callable[[Individual, tuple[int, int, int, int], random.Random, float, float], Individual]


def generate_initial_population(
    population_size: int,
    num_triangles: int,
    max_x: int,
    max_y: int,
    rng: random.Random,
) -> list[Individual]:
    """Generate an initial population of individuals, where each individual is a list of genes."""
    return [[random_triangle(max_x, max_y, rng) for _ in range(num_triangles)] for _ in range(population_size)]


def cross_and_mutate(
    parent1: Individual,
    parent2: Individual,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    mutation_rate: float,
    mutation_strength: float,
    crossover: Crossover,
    mutation_fn: MutationFn,
) -> tuple[Individual, Individual]:
    """Perform crossover and mutation to produce two child individuals."""
    child1, child2 = crossover(parent1, parent2, rng)
    child1 = mutation_fn(child1, bounds, rng, mutation_rate, mutation_strength)
    child2 = mutation_fn(child2, bounds, rng, mutation_rate, mutation_strength)
    return child1, child2


def select_parent_pair(
    parents: list[Individual],
    rng: random.Random,
) -> tuple[Individual, Individual]:
    """Pick two parents for crossover."""
    if not parents:
        raise ValueError("At least one parent is required to generate offspring")
    if len(parents) == 1:
        return parents[0], parents[0]
    parent1, parent2 = rng.sample(parents, 2)
    return parent1, parent2


def generate_offspring(
    parents: list[Individual],
    offspring_count: int,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    mutation_rate: float,
    mutation_strength: float,
    crossover: Crossover,
    mutation_fn: MutationFn,
) -> list[Individual]:
    """Generate exactly K offspring from the selected parents."""
    if offspring_count < 0:
        raise ValueError("offspring_count must be non-negative")

    offspring: list[Individual] = []
    while len(offspring) < offspring_count:
        p1, p2 = select_parent_pair(parents, rng)
        child1, child2 = cross_and_mutate(
            p1, p2, bounds, rng, mutation_rate, mutation_strength, crossover, mutation_fn
        )
        offspring.append(child1)
        if len(offspring) < offspring_count:
            offspring.append(child2)
    return offspring


def run_genetic_algorithm(
    source_image: Image.Image,
    num_triangles: int,
    population_size: int,
    generations: int,
    k: int,
    mutation_rate: float,
    mutation_strength: float,
    snapshot_interval: int,
    output_dir: str,
    selector: Selector,
    crossover: Crossover,
    fitness_fn: FitnessFn,
    survival_strategy: SurvivalStrategy,
    mutation_fn: MutationFn,
    stop_condition: StopCondition | None = None,
) -> GAResult:
    """Run the genetic algorithm and return the best result and metrics."""

    width, height = source_image.size
    source_array = np.asarray(source_image.convert("RGBA"), dtype=np.int16)
    bounds = get_overflow_bounds(width, height)
    rng = random.Random()
    population = generate_initial_population(population_size, num_triangles, width, height, rng)

    if snapshot_interval > 0:
        os.makedirs(output_dir, exist_ok=True)

    best_individual = population[0]
    best_score = float("inf")
    generations_run = generations

    for gen in range(generations):
        fitness_scores = evaluate_population_fitness(population, source_array, (width, height), fitness_fn)

        gen_best_idx = fitness_scores.index(min(fitness_scores))
        gen_best_score = fitness_scores[gen_best_idx]

        if gen_best_score < best_score:
            best_score = gen_best_score
            best_individual = population[gen_best_idx]

        print(
            f"Generation {gen + 1}/{generations} | Best fitness ({fitness_fn.__name__}): {best_score:.4f}"
        )

        if stop_condition and stop_condition(gen, best_score, fitness_scores):
            print(f"Stop condition met at generation {gen + 1}.")
            generations_run = gen + 1
            break

        if snapshot_interval > 0 and (gen + 1) % snapshot_interval == 0:
            save_phenotype_image(best_individual, output_dir, gen, width, height)

        parents = selector(population, fitness_scores, k, rng)

        offspring = generate_offspring(
            parents, k, bounds, rng, mutation_rate, mutation_strength, crossover, mutation_fn
        )
        population = survival_strategy(
            population,
            fitness_scores,
            offspring,
            source_array,
            (width, height),
            fitness_fn,
            selector,
            population_size,
            rng,
        )

    final_image = create_phenotype_image(best_individual, image_size=(width, height))
    buf = BytesIO()
    final_image.save(buf, format="PNG")
    return GAResult(image_bytes=buf.getvalue(), best_fitness=best_score, generations_run=generations_run)
