from __future__ import annotations

import os
import random
from collections.abc import Callable
from io import BytesIO

import numpy as np
from PIL import Image

from genetic_utils import Genotype, Individual, random_triangle, mutate_genotype, get_overflow_bounds
from image_utils import create_phenotype_image, compute_mae, save_phenotype_image

Selector = Callable[[list[Individual], list[float], int, random.Random], list[Individual]]
Crossover = Callable[[Individual, Individual, random.Random], tuple[Individual, Individual]]


def generate_initial_population(
    population_size: int,
    num_triangles: int,
    max_x: int,
    max_y: int,
    rng: random.Random,
) -> list[Individual]:
    """
    generate an initial population of individuals, where each individual is a list of Genotypes.
    """
    return [[random_triangle(max_x, max_y, rng) for _ in range(num_triangles)] for _ in range(population_size)]


def mutate_individual(
    individual: Individual,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    mutation_rate: float,
    mutation_strength: float,
) -> Individual:
    """Apply mutation to each triangle in an individual with the given probability."""
    return [mutate_genotype(gene, bounds, rng, mutation_strength) if rng.random() < mutation_rate else gene for gene in individual]


def cross_and_mutate(
    parent1: Individual,
    parent2: Individual,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    mutation_rate: float,
    mutation_strength: float,
    crossover: Crossover,
) -> tuple[Individual, Individual]:
    """Perform crossover and mutation to produce two child individuals."""
    child1, child2 = crossover(parent1, parent2, rng)
    child1 = mutate_individual(child1, bounds, rng, mutation_rate, mutation_strength)
    child2 = mutate_individual(child2, bounds, rng, mutation_rate, mutation_strength)
    return child1, child2


def evaluate_fitness(
    population: list[Individual],
    source_array: np.ndarray,
    image_size: tuple[int, int],
) -> list[float]:
    """Compute MAE fitness for each individual in the population."""
    return [compute_mae(source_array, create_phenotype_image(individual, image_size=image_size)) for individual in population]


def produce_offspring(
    individuals_remaining: list[Individual],
    population_size: int,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    mutation_rate: float,
    mutation_strength: float,
    crossover: Crossover,
) -> list[Individual]:
    """Generate offspring via crossover and mutation to fill the population."""
    next_population: list[Individual] = list(individuals_remaining)
    while len(next_population) < population_size:
        p1, p2 = rng.sample(individuals_remaining, 2)
        child1, child2 = cross_and_mutate(
            p1, p2, bounds, rng, mutation_rate, mutation_strength, crossover
        )
        next_population.append(child1)
        if len(next_population) < population_size:
            next_population.append(child2)
    return next_population


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
) -> bytes:
    """Run the genetic algorithm and return the best result as PNG bytes."""
    width, height = source_image.size
    source_array = np.asarray(source_image.convert("RGBA"), dtype=np.int16)
    bounds = get_overflow_bounds(width, height)
    rng = random.Random()
    population = generate_initial_population(population_size, num_triangles, width, height, rng)

    if snapshot_interval > 0:
        os.makedirs(output_dir, exist_ok=True)

    best_individual = population[0]
    best_score = float("inf")

    for gen in range(generations):
        fitness_scores = evaluate_fitness(population, source_array, (width, height))

        gen_best_idx = fitness_scores.index(min(fitness_scores))
        gen_best_score = fitness_scores[gen_best_idx]

        if gen_best_score < best_score:
            best_score = gen_best_score
            best_individual = population[gen_best_idx]

        print(f"Generation {gen + 1}/{generations} | Best fitness (MAE): {best_score:.4f}")

        if snapshot_interval > 0 and (gen + 1) % snapshot_interval == 0:
            save_phenotype_image(best_individual, output_dir, gen, width, height)

        selected = selector(population, fitness_scores, k, rng)
        population = produce_offspring(
            selected, population_size, bounds, rng, mutation_rate, mutation_strength, crossover
        )

    final_image = create_phenotype_image(best_individual, image_size=(width, height))
    buf = BytesIO()
    final_image.save(buf, format="PNG")
    return buf.getvalue()
