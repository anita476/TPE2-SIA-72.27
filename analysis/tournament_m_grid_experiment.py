"""
Tournament Size (M) vs Triangle Count Experiment

Tests tournament_deterministic with different tournament sizes (M) across
multiple triangle counts to find how optimal M scales with problem complexity.

Triangle counts tested: 20, 50, 100, 150, 200
Tournament sizes tested: 2, 3, 4, 5, 6, 8, 10, 15, 20

Outputs:
  - One results JSON per triangle count
  - A combined summary JSON for cross-triangle comparison
  - Console summary table

Usage:
  python tournament_m_triangles_experiment.py \
    --input-image input/Flag_of_Brazil.png \
    --num-runs 5 \
    --output output/tournament_m_triangles/

  # With config file
  python tournament_m_triangles_experiment.py \
    --config configs/tournament_m_triangles.json

  # Parallel execution
  python tournament_m_triangles_experiment.py \
    --input-image input/Flag_of_Brazil.png \
    --num-runs 5 \
    --max-workers 4 \
    --output output/tournament_m_triangles/
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from utils.dispatch import CROSSOVER_MAP, FITNESS_MAP, MUTATION_MAP, SURVIVAL_MAP, build_selector, build_stop_condition
from genetic_algorithm import run_genetic_algorithm
from input_output_handler import read_image


# ─────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────

def _run_single_worker(
    image_bytes: bytes,
    tournament_m: int,
    triangles: int,
    run_idx: int,
    population_size: int,
    generations: int,
    k: int,
    crossover: str,
    fitness: str,
    survival_strategy: str,
    mutation: str,
    mutation_rate: float,
    mutation_strength: float,
    convergence_window: int,
    convergence_delta: float,
    output_dir: str,
    seed: int | None = None,
) -> dict:
    """Worker: runs one GA trial for a given (triangles, M, run_idx) combination."""
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            source_image = img.convert("RGBA").copy()

        start_time = time.time()
        result = run_genetic_algorithm(
            source_image=source_image,
            num_triangles=triangles,
            population_size=population_size,
            generations=generations,
            k=k,
            selector=build_selector(
                "tournament_det",
                temperature=1.0,
                temperature_min=0.1,
                temperature_decay=-0.001,
                tournament_size=tournament_m,
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
            seed=seed,
        )
        elapsed = time.time() - start_time

        return {
            "triangles": triangles,
            "tournament_m": tournament_m,
            "run_idx": run_idx,
            "best_fitness": result.best_fitness,
            "generations_run": result.generations_run,
            "elapsed_time": elapsed,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "triangles": triangles,
            "tournament_m": tournament_m,
            "run_idx": run_idx,
            "best_fitness": None,
            "generations_run": None,
            "elapsed_time": None,
            "success": False,
            "error": str(e),
        }


# ─────────────────────────────────────────────
# Results helpers
# ─────────────────────────────────────────────

def _make_empty_results(triangle_count: int, tournament_sizes: list[int], config: dict) -> dict:
    return {
        "triangle_count": triangle_count,
        "tournament_sizes": tournament_sizes,
        "raw": {m: [] for m in tournament_sizes},
        "statistics": {
            "avg_best_fitness": {},
            "std_best_fitness": {},
            "avg_generations": {},
            "std_generations": {},
        },
        "config": config,
    }


def _compute_statistics(results: dict) -> None:
    """Fill in statistics from raw run data (in-place)."""
    for m, fitness_values in results["raw"].items():
        if fitness_values:
            results["statistics"]["avg_best_fitness"][str(m)] = float(np.mean(fitness_values))
            results["statistics"]["std_best_fitness"][str(m)] = float(np.std(fitness_values))


def _find_optimal_m(results: dict) -> tuple[int, float]:
    stats = results["statistics"]["avg_best_fitness"]
    optimal_m = max(stats, key=lambda k: stats[k])
    return int(optimal_m), stats[optimal_m]


# ─────────────────────────────────────────────
# Execution strategies
# ─────────────────────────────────────────────

def _run_sequential(
    all_results: dict[int, dict],
    tasks: list[tuple],
    total: int,
) -> None:
    start = time.time()
    for i, task in enumerate(tasks, 1):
        elapsed = time.time() - start
        eta = (elapsed / i) * (total - i) if i > 0 else 0
        print(
            f"  [{i}/{total}] triangles={task[2]:3d} M={task[1]:2d} run={task[3]+1}"
            f"  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
            end="",
        )
        worker_result = _run_single_worker(*task)
        _store_result(all_results, worker_result)


def _run_parallel(
    all_results: dict[int, dict],
    tasks: list[tuple],
    total: int,
    max_workers: int,
) -> None:
    start = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_single_worker, *task): task for task in tasks}

        for future in as_completed(future_map):
            completed += 1
            elapsed = time.time() - start
            eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
            try:
                worker_result = future.result()
                _store_result(all_results, worker_result)
                t = worker_result["triangles"]
                m = worker_result["tournament_m"]
                r = worker_result["run_idx"] + 1
                if worker_result["success"]:
                    print(
                        f"  [{completed}/{total}] triangles={t:3d} M={m:2d} run={r}"
                        f"  fitness={worker_result['best_fitness']:.6f}"
                        f"  gens={worker_result['generations_run']}"
                        f"  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m"
                    )
                else:
                    print(f"  [{completed}/{total}] triangles={t:3d} M={m:2d} run={r}  ERROR: {worker_result['error']}")
            except Exception as e:
                print(f"  [{completed}/{total}] Worker failed: {e}")


def _store_result(all_results: dict[int, dict], worker_result: dict) -> None:
    if not worker_result["success"]:
        return
    t = worker_result["triangles"]
    m = worker_result["tournament_m"]
    all_results[t]["raw"][m].append(worker_result["best_fitness"])


# ─────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────

def run_experiment(
    input_image_path: str,
    num_runs: int = 5,
    triangle_counts: list[int] | None = None,
    tournament_sizes: list[int] | None = None,
    population_size: int = 100,
    generations: int = 500,
    k: int = 60,
    crossover: str = "uniform",
    fitness: str = "rmse5",
    survival_strategy: str = "exclusive",
    mutation: str = "multigen_uniform",
    mutation_rate: float = 0.22,
    mutation_strength: float = 0.20,
    convergence_window: int = None,
    convergence_delta: float = None,
    output_dir: str = "output/tournament_m_triangles",
    max_workers: int | None = None,
    seed: int | None = None,
) -> None:

    if triangle_counts is None:
        triangle_counts = [20, 50, 100, 150, 200]
    if tournament_sizes is None:
        tournament_sizes = [2, 3, 4, 5, 6, 8, 10, 15, 20]

    os.makedirs(output_dir, exist_ok=True)

    # Load image once
    try:
        image_bytes = read_image(input_image_path)
    except (FileNotFoundError, OSError) as e:
        print(f"Error reading image: {e}")
        sys.exit(1)

    config = {
        "population_size": population_size,
        "generations": generations,
        "k": k,
        "crossover": crossover,
        "fitness": fitness,
        "survival_strategy": survival_strategy,
        "mutation": mutation,
        "mutation_rate": mutation_rate,
        "mutation_strength": mutation_strength,
        "convergence_window": convergence_window,
        "convergence_delta": convergence_delta,
        "num_runs": num_runs,
        "seed": seed,
    }

    # Build per-triangle results containers
    all_results: dict[int, dict] = {
        t: _make_empty_results(t, tournament_sizes, config)
        for t in triangle_counts
    }

    # Build flat task list for easy parallel dispatch
    tasks = []
    job_idx = 0
    for t in triangle_counts:
        for m in tournament_sizes:
            for run in range(num_runs):
                run_seed = seed + job_idx if seed is not None else None
                tasks.append((
                    image_bytes, m, t, run,
                    population_size, generations, k,
                    crossover, fitness, survival_strategy,
                    mutation, mutation_rate, mutation_strength,
                    convergence_window, convergence_delta,
                    output_dir, run_seed,
                ))
                job_idx += 1

    total = len(tasks)
    print(f"Triangle counts : {triangle_counts}")
    print(f"Tournament sizes: {tournament_sizes}")
    print(f"Runs per cell   : {num_runs}")
    print(f"Total trials    : {total}")
    print(f"Workers         : {max_workers or 'sequential'}")
    print()

    if max_workers and max_workers > 1:
        _run_parallel(all_results, tasks, total, max_workers)
    else:
        _run_sequential(all_results, tasks, total)

    # ── Compute statistics & save per-triangle JSON ──
    print("\nSaving per-triangle results...")
    for t in triangle_counts:
        _compute_statistics(all_results[t])
        per_path = Path(output_dir) / f"results_triangles_{t}.json"
        with open(per_path, "w") as f:
            json.dump(all_results[t], f, indent=2)
        print(f"  Saved: {per_path}")

    # ── Build combined summary ──
    summary = {
        "triangle_counts": triangle_counts,
        "tournament_sizes": tournament_sizes,
        "config": config,
        # optimal_m[str(t)] = {"optimal_m": int, "fitness": float}
        "optimal_m_per_triangle": {},
        # avg_fitness[str(t)][str(m)] = float
        "avg_fitness": {},
    }

    for t in triangle_counts:
        opt_m, opt_fit = _find_optimal_m(all_results[t])
        summary["optimal_m_per_triangle"][str(t)] = {
            "optimal_m": opt_m,
            "fitness": opt_fit,
        }
        summary["avg_fitness"][str(t)] = all_results[t]["statistics"]["avg_best_fitness"]

    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCombined summary saved: {summary_path}")

    # ── Print console summary table ──
    _print_summary(summary, triangle_counts, tournament_sizes)


def _print_summary(summary: dict, triangle_counts: list[int], tournament_sizes: list[int]) -> None:
    print("\n" + "=" * 90)
    print("SUMMARY: Average Best Fitness  (rows = triangles, cols = M)")
    print("=" * 90)

    # Header
    header = f"{'Triangles':>10s}" + "".join(f"  M={m:<4d}" for m in tournament_sizes) + "  │  Optimal M"
    print(header)
    print("-" * len(header))

    for t in triangle_counts:
        avg = summary["avg_fitness"].get(str(t), {})
        opt = summary["optimal_m_per_triangle"].get(str(t), {})
        row = f"{t:>10d}"
        for m in tournament_sizes:
            val = avg.get(str(m), float("nan"))
            row += f"  {val:>6.4f}"
        row += f"  │  M={opt.get('optimal_m', '?')}  (fitness={opt.get('fitness', float('nan')):.4f})"
        print(row)

    print("=" * 90)
    print("\nOptimal M scaling:")
    for t in triangle_counts:
        opt = summary["optimal_m_per_triangle"].get(str(t), {})
        print(f"  {t:3d} triangles  →  optimal M = {opt.get('optimal_m', '?')}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse config JSON: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--input-image", type=str)
    parser.add_argument("--num-runs", type=int)
    parser.add_argument("--triangle-counts", type=int, nargs="+", help="Triangle counts to sweep (default: 20 50 100 150 200)")
    parser.add_argument("--tournament-sizes", type=int, nargs="+", help="M values to test (default: 2 3 4 5 6 8 10 15 20)")
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--crossover", type=str, choices=list(CROSSOVER_MAP.keys()))
    parser.add_argument("--fitness", type=str, choices=list(FITNESS_MAP.keys()))
    parser.add_argument("--mutation", type=str, choices=list(MUTATION_MAP.keys()))
    parser.add_argument("--mutation-rate", type=float)
    parser.add_argument("--mutation-strength", type=float)
    parser.add_argument("--convergence-window", type=int)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}

    input_image = args.input_image or cfg.get("input_image")
    if not input_image:
        print("Error: --input-image is required (via CLI or config)")
        sys.exit(1)

    run_experiment(
        input_image_path=input_image,
        num_runs=args.num_runs or cfg.get("num_runs", 5),
        triangle_counts=args.triangle_counts or cfg.get("triangle_counts", [200]),
        tournament_sizes=args.tournament_sizes or cfg.get("tournament_sizes", [2, 3, 4, 5, 6, 8, 10, 15, 20]),
        population_size=args.population_size or cfg.get("population_size", 100),
        generations=args.generations or cfg.get("generations", 1000),
        k=args.k or cfg.get("k", 40),
        crossover=args.crossover or cfg.get("crossover", "annular"),
        fitness=args.fitness or cfg.get("fitness", "rmse5"),
        survival_strategy=cfg.get("survival_strategy", "exclusive"),
        mutation=args.mutation or cfg.get("mutation", "multigen_limited"),
        mutation_rate=args.mutation_rate or cfg.get("mutation_rate", 0.22),
        mutation_strength=args.mutation_strength or cfg.get("mutation_strength", 0.28),
        convergence_window=args.convergence_window or cfg.get("convergence_window", None),
        convergence_delta=cfg.get("convergence_delta", None),
        output_dir=args.output or cfg.get("output", "output/tournament_m_triangles"),
        max_workers=args.max_workers or cfg.get("max_workers"),
        seed=args.seed or cfg.get("seed"),
    )


if __name__ == "__main__":
    main()