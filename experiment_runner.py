"""
Runs a grid search over hyperparameter combinations and saves results to a CSV.

Grid config format (JSON):
{
  "base": {
    "input_image": "input/Flag_of_France.svg.png",
    "triangles": 200,
    "mutation_rate": 0.1,
    "mutation_strength": 0.3,
    "snapshot_interval": 0,
    "output_dir": "output/experiments"
  },
  "grid": {
    "selector":          ["elite", "roulette", "boltzmann"],
    "crossover":         ["one_point", "two_point"],
    "fitness":           ["mae"],
    "survival_strategy": ["additive", "exclusive"],
    "population_size":   [50, 100],
    "generations":       [500],
    "k":                 [20, 50]
  }
}

Usage:
  python3 experiment_runner.py --config configs/experiment_grid.json --output results.csv
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
from io import BytesIO

from PIL import Image

from utils.dispatch import CROSSOVER_MAP, FITNESS_MAP, MUTATION_MAP, SURVIVAL_MAP, build_selector, build_stop_condition
from genetic_algorithm import run_genetic_algorithm
from input_output_handler import read_image


def run_experiments(grid_config_path: str, output_csv: str, save_images: bool) -> None:
    with open(grid_config_path) as f:
        config = json.load(f)

    base = config["base"]
    grid = config["grid"]

    image_bytes = read_image(base["input_image"])
    with Image.open(BytesIO(image_bytes)) as img:
        source_image = img.convert("RGBA").copy()

    keys = list(grid.keys())
    combinations = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combinations)
    print(f"Running {total} experiment(s)...\n")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = keys + ["best_fitness", "generations_run", "elapsed_seconds"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, values in enumerate(combinations):
            params = dict(zip(keys, values))
            merged = {**base, **params}
            print(f"[{i + 1}/{total}] {params}")

            t0 = time.perf_counter()
            result = run_genetic_algorithm(
                source_image=source_image,
                num_triangles=merged["triangles"],
                population_size=merged["population_size"],
                generations=merged["generations"],
                k=merged["k"],
                selector=build_selector(
                    merged["selector"],
                    merged.get("temperature", 50.0),
                    merged.get("temperature_min", 1.0),
                    merged.get("temperature_decay", -0.005),
                    merged.get("tournament_threshold", 0.5),
                ),
                crossover=CROSSOVER_MAP[merged["crossover"]],
                fitness_fn=FITNESS_MAP[merged["fitness"]],
                survival_strategy=SURVIVAL_MAP[merged["survival_strategy"]],
                mutation_fn=MUTATION_MAP[merged.get("mutation", "multigen_uniform")],
                mutation_rate=merged.get("mutation_rate", 0.1),
                mutation_strength=merged.get("mutation_strength", 0.3),
                snapshot_interval=0,
                output_dir=merged.get("output_dir", "output/experiments"),
                stop_condition=build_stop_condition(
                    merged.get("target_fitness"),
                    merged.get("convergence_window"),
                    merged.get("convergence_delta", 1e-4),
                    merged.get("time_limit"),
                ),
            )
            elapsed = round(time.perf_counter() - t0, 4)

            row = {
                **params,
                "best_fitness": result.best_fitness,
                "generations_run": result.generations_run,
                "elapsed_seconds": elapsed,
            }
            writer.writerow(row)
            csvfile.flush()

            if save_images:
                label = "_".join(f"{k}{v}" for k, v in params.items())
                image_path = os.path.join(merged.get("output_dir", "output/experiments"), f"{label}.png")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                with open(image_path, "wb") as f:
                    f.write(result.image_bytes)

            print(
                f"    best_fitness={result.best_fitness:.4f}  "
                f"generations_run={result.generations_run}  "
                f"elapsed={elapsed:.2f}s\n"
            )

    print(f"Results saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Grid search over GA hyperparameters.")
    parser.add_argument("--config", required=True, help="Path to experiment grid JSON config")
    parser.add_argument("--output", default="results.csv", help="Path to output CSV file")
    parser.add_argument("--save-images", action="store_true", help="Save output image for each combination")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    run_experiments(args.config, args.output, args.save_images)


if __name__ == "__main__":
    main()
