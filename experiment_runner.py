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
import concurrent.futures
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


def _history_csv_path(output_csv: str) -> str:
    base, _ = os.path.splitext(output_csv)
    return f"{base}_history.csv"


def _run_one(image_bytes: bytes, merged: dict, params: dict, verbose: bool = False) -> dict:
    """Run a single GA experiment. Designed to be called in a subprocess."""
    with Image.open(BytesIO(image_bytes)) as img:
        source_image = img.convert("RGBA").copy()

    t0 = time.perf_counter()
    result = run_genetic_algorithm(
        source_image=source_image,
        num_triangles=merged["triangles"],
        population_size=merged["population_size"],
        generations=merged["generations"],
        k=merged["k"],
        selector=build_selector(
            merged["selector"],
            merged.get("temperature", 1.0),
            merged.get("temperature_min", 0.1),
            merged.get("temperature_decay", -0.001),
            merged.get("tournament_threshold", 0.75),
        ),
        crossover=CROSSOVER_MAP[merged["crossover"]],
        fitness_fn=FITNESS_MAP[merged["fitness"]],
        survival_strategy=SURVIVAL_MAP[merged["survival_strategy"]],
        mutation_fn=MUTATION_MAP[merged.get("mutation", "multigen_limited")],
        mutation_rate=merged.get("mutation_rate", 0.15),
        mutation_strength=merged.get("mutation_strength", 0.15),
        snapshot_interval=0,
        output_dir=merged.get("output_dir", "output/experiments"),
        seed=merged.get("seed"),
        stop_condition=build_stop_condition(
            merged.get("target_fitness"),
            merged.get("convergence_window"),
            merged.get("convergence_delta", 1e-4),
            merged.get("time_limit"),
        ),
        verbose=verbose,
    )
    elapsed = round(time.perf_counter() - t0, 4)
    return {
        "params": params,
        "merged": merged,
        "best_fitness": result.best_fitness,
        "generations_run": result.generations_run,
        "elapsed_seconds": elapsed,
        "fitness_history": result.fitness_history,
        "time_history": result.time_history,
        "image_bytes": result.image_bytes,
    }


def run_experiments(
    grid_config_path: str,
    output_csv: str,
    save_images: bool,
    workers: int = 1,
) -> None:
    with open(grid_config_path) as f:
        config = json.load(f)

    base = config["base"]
    grid = config.get("grid", {})
    pairs = config.get("pairs", [])

    image_bytes = read_image(base["input_image"])

    keys = list(grid.keys())
    grid_combinations = list(itertools.product(*[grid[k] for k in keys]))

    if pairs:
        pair_keys_list = [list(p.keys()) for p in pairs]
        pair_vals_list = [list(zip(*p.values())) for p in pairs]
        pair_combinations = list(itertools.product(*pair_vals_list))

        flat_pair_keys = [k for group in pair_keys_list for k in group]
        combinations = []
        for grid_vals in grid_combinations:
            for pair_vals_tuple in pair_combinations:
                flat_pair_vals = [v for group in pair_vals_tuple for v in group]
                combinations.append(tuple(list(grid_vals) + flat_pair_vals))
        keys = keys + flat_pair_keys
    else:
        combinations = grid_combinations
    total = len(combinations)
    workers = min(workers, total)
    print(f"Running {total} experiment(s) with {workers} worker(s)...\n")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    history_csv = _history_csv_path(output_csv)
    fieldnames = keys + ["best_fitness", "generations_run", "elapsed_seconds"]
    history_fieldnames = keys + ["generation", "best_fitness", "elapsed_seconds"]

    with open(output_csv, "w", newline="") as csvfile, \
         open(history_csv, "w", newline="") as histfile:

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        hist_writer = csv.DictWriter(histfile, fieldnames=history_fieldnames)
        hist_writer.writeheader()

        jobs = []
        for values in combinations:
            params = dict(zip(keys, values))
            merged = {**base, **params}
            jobs.append((params, merged))

        executor_cls = (
            concurrent.futures.ProcessPoolExecutor if workers > 1
            else concurrent.futures.ThreadPoolExecutor
        )
        verbose_run = workers == 1
        with executor_cls(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one, image_bytes, merged, params, verbose_run): (i, (params, merged))
                for i, (params, merged) in enumerate(jobs)
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                i, (params, _) = futures[future]
                completed += 1
                try:
                    r = future.result()
                except Exception as exc:
                    print(f"[{i + 1}/{total}] ERROR: {params} → {exc}")
                    continue

                print(
                    f"[{completed}/{total}] {r['params']}  "
                    f"best_fitness={r['best_fitness']:.4f}  "
                    f"generations_run={r['generations_run']}  "
                    f"elapsed={r['elapsed_seconds']:.2f}s"
                )

                writer.writerow({
                    **r["params"],
                    "best_fitness": r["best_fitness"],
                    "generations_run": r["generations_run"],
                    "elapsed_seconds": r["elapsed_seconds"],
                })
                csvfile.flush()

                for gen_idx, (fitness_val, time_val) in enumerate(
                    zip(r["fitness_history"], r["time_history"])
                ):
                    hist_writer.writerow({
                        **r["params"],
                        "generation": gen_idx + 1,
                        "best_fitness": fitness_val,
                        "elapsed_seconds": time_val,
                    })
                histfile.flush()

                if save_images:
                    label = "_".join(f"{k}{v}" for k, v in r["params"].items())
                    out_dir = r["merged"].get("output_dir", "output/experiments")
                    image_path = os.path.join(out_dir, f"{label}.png")
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    with open(image_path, "wb") as f:
                        f.write(r["image_bytes"])

    print(f"\nResults saved to {output_csv}")
    print(f"History  saved to {history_csv}")


def main():
    parser = argparse.ArgumentParser(description="Grid search over GA hyperparameters.")
    parser.add_argument("--config", required=True, help="Path to experiment grid JSON config")
    parser.add_argument("--output", default="results.csv", help="Path to output CSV file")
    parser.add_argument("--save-images", action="store_true", help="Save output image for each combination")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1). "
             "Use --workers 0 to use all available CPU cores.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    workers = os.cpu_count() if args.workers == 0 else args.workers
    run_experiments(args.config, args.output, args.save_images, workers=workers)


if __name__ == "__main__":
    main()
