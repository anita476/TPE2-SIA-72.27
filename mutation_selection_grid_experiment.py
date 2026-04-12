"""
Mutation Operator × Selection Method Grid Experiment

Runs all combinations of mutation operators and selection methods to identify:
1. Which mutation works best overall
2. Synergistic selection+mutation combinations  
3. Whether high-pressure selection pairs better with conservative mutation
4. Whether low-pressure selection pairs better with aggressive mutation

Outputs results as JSON for heatmap visualization.

Usage:
  python mutation_selection_grid_experiment.py \
    --input-image input/Flag_of_France.svg.png \
    --num-runs 5 \
    --output output/mutation_selection_grid/results.json
"""

import argparse
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from utils.dispatch import CROSSOVER_MAP, FITNESS_MAP, MUTATION_MAP, SURVIVAL_MAP, build_selector, build_stop_condition
from genetic_algorithm import run_genetic_algorithm
from input_output_handler import read_image


def run_grid_experiment(
    input_image_path: str,
    num_runs: int,
    triangles: int = 200,
    population_size: int = 100,
    generations: int = 500,
    k: int = 40,
    crossover: str = "annular",
    fitness: str = "rmse5",
    survival_strategy: str = "exclusive",
    mutation_rate: float = 0.22,
    mutation_strength: float = 0.28,
    convergence_window: int | None = 20,
    convergence_delta: float = 1e-4,
    output_dir: str = "output/mutation_selection_grid",
) -> None:
    """Run all mutation × selection combinations."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    try:
        image_bytes = read_image(input_image_path)
    except (FileNotFoundError, OSError) as e:
        print(f"Error reading image: {e}")
        sys.exit(1)

    with Image.open(BytesIO(image_bytes)) as img:
        source_image = img.convert("RGBA").copy()

    # Define grid
    mutations = list(MUTATION_MAP.keys())
    selections = [
        "elite",
        "tournament_det",
        "tournament_stoch",
        "roulette",
        "ranking",
        "boltzmann",
        "universal",
    ]
    
    # Results storage
    results = {
        "mutations": mutations,
        "selections": selections,
        "results": {selection: {mutation: [] for mutation in mutations} for selection in selections},
        "config": {
            "triangles": triangles,
            "population_size": population_size,
            "generations": generations,
            "k": k,
            "crossover": crossover,
            "fitness": fitness,
            "survival_strategy": survival_strategy,
            "mutation_rate": mutation_rate,
            "mutation_strength": mutation_strength,
            "convergence_window": convergence_window,
            "convergence_delta": convergence_delta,
            "num_runs": num_runs,
        },
        "statistics": {
            "avg_best_fitness": {},
            "std_best_fitness": {},
            "avg_worst_fitness": {},
            "std_worst_fitness": {},
            "avg_generations": {},
            "std_generations": {},
        }
    }
    
    total_experiments = len(mutations) * len(selections) * num_runs
    experiment_count = 0
    
    print(f"Running mutation × selection grid experiment")
    print(f"Mutations: {len(mutations)}, Selections: {len(selections)}, Runs per combo: {num_runs}")
    print(f"Total experiments: {total_experiments}\n")
    
    start_time_total = time.time()
    
    # Run all combinations
    for sel_idx, selection in enumerate(selections):
        print(f"\n{'='*70}")
        print(f"Selection Method: {selection} ({sel_idx+1}/{len(selections)})")
        print(f"{'='*70}\n")
        
        for mut_idx, mutation in enumerate(mutations):
            print(f"Mutation: {mutation:20s} ({mut_idx+1}/{len(mutations)})")
            
            mutation_results = {
                "best_fitness": [],
                "worst_fitness": [],
                "generations": [],
                "execution_time": [],
            }
            
            for run in range(num_runs):
                experiment_count += 1
                elapsed_total = time.time() - start_time_total
                eta_remaining = (elapsed_total / experiment_count) * (total_experiments - experiment_count)
                
                print(f"  Run {run+1}/{num_runs}  "
                      f"[{experiment_count}/{total_experiments}]  "
                      f"Elapsed: {elapsed_total/60:.1f}m  "
                      f"ETA: {eta_remaining/60:.1f}m", end="")
                
                try:
                    run_start = time.time()
                    result = run_genetic_algorithm(
                        source_image=source_image,
                        num_triangles=triangles,
                        population_size=population_size,
                        generations=generations,
                        k=k,
                        selector=build_selector(
                            selection,
                            temperature=1.0,
                            temperature_min=0.1,
                            temperature_decay=-0.001,
                            tournament_threshold=0.75,
                        ),
                        crossover=CROSSOVER_MAP[crossover],
                        fitness_fn=FITNESS_MAP[fitness],
                        survival_strategy=SURVIVAL_MAP[survival_strategy],
                        mutation_fn=MUTATION_MAP[mutation],
                        mutation_rate=mutation_rate,
                        mutation_strength=mutation_strength,
                        snapshot_interval=0,
                        output_dir=output_dir,
                        stop_condition=build_stop_condition(
                            target_fitness_val=None,
                            convergence_window=convergence_window,
                            convergence_delta=convergence_delta,
                            time_limit_secs=None,
                        ),
                    )
                    run_elapsed = time.time() - run_start
                    
                    mutation_results["best_fitness"].append(result.best_fitness)
                    mutation_results["worst_fitness"].append(result.worst_fitness)
                    mutation_results["generations"].append(result.generations_run)
                    mutation_results["execution_time"].append(run_elapsed)
                    
                    print(f"  → fitness={result.best_fitness:.6f}  gens={result.generations_run}")
                    
                except Exception as e:
                    print(f"  → ERROR: {e}")
                    continue
            
            # Store results
            results["results"][selection][mutation] = mutation_results["best_fitness"]
    
    # Compute statistics
    for selection in selections:
        results["statistics"]["avg_best_fitness"][selection] = {}
        results["statistics"]["std_best_fitness"][selection] = {}
        
        for mutation in mutations:
            fitness_values = results["results"][selection][mutation]
            if fitness_values:
                results["statistics"]["avg_best_fitness"][selection][mutation] = float(np.mean(fitness_values))
                results["statistics"]["std_best_fitness"][selection][mutation] = float(np.std(fitness_values))
    
    # Save results
    results_path = Path(output_dir) / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    elapsed_total = time.time() - start_time_total
    print(f"\n{'='*70}")
    print(f"Experiment complete in {elapsed_total/60:.1f} minutes")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}\n")
    
    # Print summary table
    print("SUMMARY: Average Best Fitness")
    print(f"\n{'Selection':<20}", end="")
    for mutation in mutations:
        print(f"{mutation:16s}", end="")
    print()
    print("-" * (20 + len(mutations) * 16))
    
    for selection in selections:
        print(f"{selection:<20}", end="")
        for mutation in mutations:
            avg = results["statistics"]["avg_best_fitness"][selection][mutation]
            print(f"{avg:16.6f}", end="")
        print()


def load_config(config_path: str) -> dict:
    """Load experiment configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse config JSON: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file (values will be used as defaults, "
             "overridden by CLI arguments)",
    )
    
    parser.add_argument(
        "--input-image",
        type=str,
        help="Path to input image",
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        help="Number of runs per combination [default: 5]",
    )
    
    parser.add_argument(
        "--triangles",
        type=int,
        help="Number of triangles [default: 200]",
    )
    
    parser.add_argument(
        "--population-size",
        type=int,
        help="Population size [default: 100]",
    )
    
    parser.add_argument(
        "--generations",
        type=int,
        help="Max generations [default: 500]",
    )
    
    parser.add_argument(
        "--k",
        type=int,
        help="Selection pool size [default: 40]",
    )
    
    parser.add_argument(
        "--crossover",
        type=str,
        choices=list(CROSSOVER_MAP.keys()),
        help="Crossover operator [default: annular]",
    )
    
    parser.add_argument(
        "--fitness",
        type=str,
        choices=list(FITNESS_MAP.keys()),
        help="Fitness function [default: rmse5]",
    )
    
    parser.add_argument(
        "--survival-strategy",
        type=str,
        choices=list(SURVIVAL_MAP.keys()),
        help="Survival strategy [default: exclusive]",
    )
    
    parser.add_argument(
        "--mutation-rate",
        type=float,
        help="Mutation rate [default: 0.22]",
    )
    
    parser.add_argument(
        "--mutation-strength",
        type=float,
        help="Mutation strength [default: 0.28]",
    )
    
    parser.add_argument(
        "--convergence-window",
        type=int,
        help="Convergence window for early stopping [default: 20]",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory [default: output/mutation_selection_grid]",
    )
    
    args = parser.parse_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}\n")
    
    # CLI arguments override config file values
    input_image = args.input_image or config.get("input_image")
    if not input_image:
        print("Error: --input-image is required (via CLI or config)")
        sys.exit(1)
    
    run_grid_experiment(
        input_image_path=input_image,
        num_runs=args.num_runs or config.get("num_runs", 5),
        triangles=args.triangles or config.get("triangles", 200),
        population_size=args.population_size or config.get("population_size", 100),
        generations=args.generations or config.get("generations", 500),
        k=args.k or config.get("k", 40),
        crossover=args.crossover or config.get("crossover", "annular"),
        fitness=args.fitness or config.get("fitness", "rmse5"),
        survival_strategy=args.survival_strategy or config.get("survival_strategy", "exclusive"),
        mutation_rate=args.mutation_rate or config.get("mutation_rate", 0.22),
        mutation_strength=args.mutation_strength or config.get("mutation_strength", 0.28),
        convergence_window=args.convergence_window or config.get("convergence_window", 20),
        output_dir=args.output or config.get("output", "output/mutation_selection_grid"),
    )


if __name__ == "__main__":
    main()
