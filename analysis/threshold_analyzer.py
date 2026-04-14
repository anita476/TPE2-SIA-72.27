"""
Analyzes the effect of stochastic tournament threshold on GA performance.

Runs the GA multiple times with uniformly-spaced thresholds from 0.5 to 1.0,
using stochastic tournament selection and no_improvement stop condition.

Supports seeded reproducibility: use --seed to ensure identical runs across experiments.

Plots:
1. Final best fitness score vs threshold
2. Worst final fitness score vs threshold
3. Number of generations vs threshold
4. Average execution time vs threshold
5. Comparison with elite and tournament deterministic selections

Usage:
  # Basic usage
  python threshold_analyzer.py --input-image input/Flag_of_France.svg.png \
                               --num-runs 5 \
                               --num-thresholds 10 \
                               --convergence-window 20
  
  # With reproducible seed (each run gets unique seed: seed+0, seed+1, seed+2, ...)
  python threshold_analyzer.py --input-image input/Flag_of_France.svg.png \
                               --seed 42 \
                               --num-runs 5
  
  # From config file
  python threshold_analyzer.py --config configs/threshold_analysis.json
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.dispatch import CROSSOVER_MAP, FITNESS_MAP, MUTATION_MAP, SURVIVAL_MAP, build_selector, build_stop_condition
from genetic_algorithm import run_genetic_algorithm
from input_output_handler import read_image, save_image


def _run_one_threshold(image_bytes: bytes, job: dict, verbose: bool = False) -> dict:
    """Run a single GA experiment for threshold analysis. Module-level for pickling."""
    with Image.open(BytesIO(image_bytes)) as img:
        source_image = img.convert("RGBA").copy()

    selector_name = job["selector_name"]
    threshold = job.get("threshold", 0.75)

    selector = build_selector(
        selector_name,
        temperature=1.0,
        temperature_min=0.1,
        temperature_decay=-0.001,
        tournament_threshold=threshold,
    )
    stop = build_stop_condition(
        target_fitness_val=None,
        convergence_window=job.get("convergence_window"),
        convergence_delta=job.get("convergence_delta", 1e-4),
        time_limit_secs=None,
    )

    t0 = time.perf_counter()
    result = run_genetic_algorithm(
        source_image=source_image,
        num_triangles=job["triangles"],
        population_size=job["population_size"],
        generations=job["generations"],
        k=job["k"],
        selector=selector,
        crossover=CROSSOVER_MAP[job["crossover"]],
        fitness_fn=FITNESS_MAP[job["fitness"]],
        survival_strategy=SURVIVAL_MAP[job["survival_strategy"]],
        mutation_fn=MUTATION_MAP[job["mutation"]],
        mutation_rate=job["mutation_rate"],
        mutation_strength=job["mutation_strength"],
        snapshot_interval=0,
        output_dir=job["output_dir"],
        stop_condition=stop,
        verbose=verbose,
        seed=job.get("seed"),
    )
    elapsed = time.perf_counter() - t0

    return {
        "selector_name": selector_name,
        "threshold": threshold,
        "run_index": job["run_index"],
        "best_fitness": result.best_fitness,
        "generations_run": result.generations_run,
        "elapsed_seconds": elapsed,
        "image_bytes": result.image_bytes,
    }


def run_threshold_analysis(
    input_image_path: str,
    num_runs: int,
    num_thresholds: int,
    triangles: int = 200,
    population_size: int = 100,
    generations: int = 500,
    k: int = 40,
    crossover: str = "annular",
    fitness: str = "rmse5",
    survival_strategy: str = "exclusive",
    mutation: str = "multigen_limited",
    mutation_rate: float = 0.22,
    mutation_strength: float = 0.28,
    convergence_window: int | None = None,
    convergence_delta: float = 1e-4,
    output_dir: str = "output/threshold_analysis",
    workers: int = 1,
    verbose: bool = False,
    seed: int | None = None,
) -> None:
    """Run GA experiments with varying stochastic tournament thresholds."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "final_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Load image
    try:
        image_bytes = read_image(input_image_path)
    except (FileNotFoundError, OSError) as e:
        print(f"Error reading image: {e}")
        sys.exit(1)

    thresholds = np.linspace(0.5, 1.0, num_thresholds)

    base_params = dict(
        triangles=triangles,
        population_size=population_size,
        generations=generations,
        k=k,
        crossover=crossover,
        fitness=fitness,
        survival_strategy=survival_strategy,
        mutation=mutation,
        mutation_rate=mutation_rate,
        mutation_strength=mutation_strength,
        convergence_window=convergence_window,
        convergence_delta=convergence_delta,
        output_dir=output_dir,
    )

    # Build job list
    jobs: list[dict] = []
    job_counter = 0

    for baseline_name in ["elite", "tournament_det"]:
        for run in range(num_runs):
            job_seed = seed + job_counter if seed is not None else None
            jobs.append({**base_params, "selector_name": baseline_name,
                         "threshold": 0.75, "run_index": run,
                         "convergence_window": None, "seed": job_seed})  # baselines run to completion
            job_counter += 1

    for threshold in thresholds:
        for run in range(num_runs):
            job_seed = seed + job_counter if seed is not None else None
            jobs.append({**base_params, "selector_name": "tournament_stoch",
                         "threshold": float(threshold), "run_index": run, "seed": job_seed})
            job_counter += 1


    total = len(jobs)
    print(f"Running {total} experiments "
          f"({num_runs} runs × {num_thresholds} thresholds + {num_runs} × 2 baselines) "
          f"with {workers} worker(s)...\n")

    verbose_run = verbose
    executor_cls = (
        concurrent.futures.ProcessPoolExecutor if workers > 1
        else concurrent.futures.ThreadPoolExecutor
    )

    raw: list[dict] = []

    with executor_cls(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(_run_one_threshold, image_bytes, job, verbose_run): job
            for job in jobs
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_job):
            completed += 1
            job = future_to_job[future]
            try:
                res = future.result()
                raw.append(res)
                print(
                    f"[{completed}/{total}] {res['selector_name']} "
                    f"thr={res['threshold']:.3f} run={res['run_index']+1} "
                    f"fitness={res['best_fitness']:.6f} "
                    f"gens={res['generations_run']} "
                    f"time={res['elapsed_seconds']:.2f}s"
                )
            except Exception as e:
                print(f"[{completed}/{total}] ERROR {job['selector_name']} "
                      f"thr={job.get('threshold', 0):.3f} run={job['run_index']+1}: {e}")

    # Aggregate baseline results
    baseline_results: dict[str, dict] = {
        "elite": {"best_fitness": [], "generations": [], "execution_time": []},
        "tournament_det": {"best_fitness": [], "generations": [], "execution_time": []},
    }
    for r in raw:
        name = r["selector_name"]
        if name in baseline_results:
            baseline_results[name]["best_fitness"].append(r["best_fitness"])
            baseline_results[name]["generations"].append(r["generations_run"])
            baseline_results[name]["execution_time"].append(r["elapsed_seconds"])

    # Aggregate stochastic threshold results
    thr_buckets: dict[float, dict] = {}
    for r in raw:
        if r["selector_name"] != "tournament_stoch":
            continue
        thr = r["threshold"]
        if thr not in thr_buckets:
            thr_buckets[thr] = {"best_fitness": [], "generations": [], "execution_time": []}
        thr_buckets[thr]["best_fitness"].append(r["best_fitness"])
        thr_buckets[thr]["generations"].append(r["generations_run"])
        thr_buckets[thr]["execution_time"].append(r["elapsed_seconds"])

    # Save first-run images per threshold
    saved_thr: set[float] = set()
    for r in sorted(raw, key=lambda x: x["run_index"]):
        if r["selector_name"] == "tournament_stoch" and r["threshold"] not in saved_thr:
            img_path = os.path.join(images_dir, f"threshold_{r['threshold']:.3f}_final.png")
            save_image(r["image_bytes"], img_path)
            saved_thr.add(r["threshold"])

    results: dict = {
        "thresholds": [],
        "best_fitness_scores": [],
        "worst_fitness_scores": [],
        "generations_run": [],
        "execution_times": [],
        "avg_best_fitness": [],
        "std_best_fitness": [],
        "avg_worst_fitness": [],
        "std_worst_fitness": [],
        "avg_generations": [],
        "std_generations": [],
        "avg_execution_time": [],
        "std_execution_time": [],
    }

    for thr in sorted(thr_buckets):
        b = thr_buckets[thr]
        if not b["best_fitness"]:
            continue
        results["thresholds"].append(thr)
        results["best_fitness_scores"].append(b["best_fitness"])
        results["worst_fitness_scores"].append(b["best_fitness"])
        results["generations_run"].append(b["generations"])
        results["execution_times"].append(b["execution_time"])
        results["avg_best_fitness"].append(float(np.mean(b["best_fitness"])))
        results["std_best_fitness"].append(float(np.std(b["best_fitness"])))
        results["avg_worst_fitness"].append(float(np.mean(b["best_fitness"])))
        results["std_worst_fitness"].append(float(np.std(b["best_fitness"])))
        results["avg_generations"].append(float(np.mean(b["generations"])))
        results["std_generations"].append(float(np.std(b["generations"])))
        results["avg_execution_time"].append(float(np.mean(b["execution_time"])))
        results["std_execution_time"].append(float(np.std(b["execution_time"])))

    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump({
            "thresholds": results["thresholds"],
            "avg_best_fitness": results["avg_best_fitness"],
            "std_best_fitness": results["std_best_fitness"],
            "avg_worst_fitness": results["avg_worst_fitness"],
            "std_worst_fitness": results["std_worst_fitness"],
            "avg_generations": results["avg_generations"],
            "std_generations": results["std_generations"],
            "avg_execution_time": results["avg_execution_time"],
            "std_execution_time": results["std_execution_time"],
            "baseline_elite_fitness": float(np.mean(baseline_results["elite"]["best_fitness"])) if baseline_results["elite"]["best_fitness"] else 0,
            "baseline_elite_generations": float(np.mean(baseline_results["elite"]["generations"])) if baseline_results["elite"]["generations"] else 0,
            "baseline_elite_time": float(np.mean(baseline_results["elite"]["execution_time"])) if baseline_results["elite"]["execution_time"] else 0,
            "baseline_tournament_det_fitness": float(np.mean(baseline_results["tournament_det"]["best_fitness"])) if baseline_results["tournament_det"]["best_fitness"] else 0,
            "baseline_tournament_det_generations": float(np.mean(baseline_results["tournament_det"]["generations"])) if baseline_results["tournament_det"]["generations"] else 0,
            "baseline_tournament_det_time": float(np.mean(baseline_results["tournament_det"]["execution_time"])) if baseline_results["tournament_det"]["execution_time"] else 0,
        }, f, indent=2)
    print(f"\nResults saved to {results_json_path}")
    print(f"Final images saved to {images_dir}\n")

    create_plots(results, baseline_results, output_dir)


def create_plots(results: dict, baseline_results: dict, output_dir: str) -> None:
    """Create and save plots of the analysis results."""

    if not results["thresholds"]:
        print("No results to plot")
        return

    thresholds = np.array(results["thresholds"])
    avg_best_fitness = np.array(results["avg_best_fitness"])
    std_best_fitness = np.array(results["std_best_fitness"])
    avg_worst_fitness = np.array(results["avg_worst_fitness"])
    std_worst_fitness = np.array(results["std_worst_fitness"])
    avg_generations = np.array(results["avg_generations"])
    std_generations = np.array(results["std_generations"])
    avg_execution_time = np.array(results["avg_execution_time"])
    std_execution_time = np.array(results["std_execution_time"])

    elite_fitness = np.mean(baseline_results["elite"]["best_fitness"]) if baseline_results["elite"]["best_fitness"] else 0
    elite_generations = np.mean(baseline_results["elite"]["generations"]) if baseline_results["elite"]["generations"] else 0
    elite_time = np.mean(baseline_results["elite"]["execution_time"]) if baseline_results["elite"]["execution_time"] else 0

    tournament_det_fitness = np.mean(baseline_results["tournament_det"]["best_fitness"]) if baseline_results["tournament_det"]["best_fitness"] else 0
    tournament_det_generations = np.mean(baseline_results["tournament_det"]["generations"]) if baseline_results["tournament_det"]["generations"] else 0
    tournament_det_time = np.mean(baseline_results["tournament_det"]["execution_time"]) if baseline_results["tournament_det"]["execution_time"] else 0

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].errorbar(thresholds, avg_best_fitness, yerr=std_best_fitness,
                        marker='o', linewidth=2.5, markersize=8, capsize=5, capthick=2,
                        color='#2E86AB', label='Stochastic Tournament', ecolor='#2E86AB', alpha=0.7)
    axes[0, 0].axhline(y=elite_fitness, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite ({elite_fitness:.4f})')
    axes[0, 0].axhline(y=tournament_det_fitness, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det ({tournament_det_fitness:.4f})')
    axes[0, 0].fill_between(thresholds, avg_best_fitness - std_best_fitness, avg_best_fitness + std_best_fitness, alpha=0.2, color='#2E86AB')
    for i, (x, y) in enumerate(zip(thresholds, avg_best_fitness)):
        axes[0, 0].text(x, y + std_best_fitness[i] + 0.02, f'{y:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0, 0].set_xlabel('Tournament Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Average Best Fitness', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Final Best Fitness vs Threshold', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0.48, 1.02)
    axes[0, 0].legend(loc='lower right', fontsize=10)

    axes[0, 1].errorbar(thresholds, avg_generations, yerr=std_generations,
                        marker='^', linewidth=2.5, markersize=8, capsize=5, capthick=2,
                        color='#F18F01', label='Stochastic Tournament', ecolor='#F18F01', alpha=0.7)
    axes[0, 1].axhline(y=elite_generations, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite ({elite_generations:.1f})')
    axes[0, 1].axhline(y=tournament_det_generations, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det ({tournament_det_generations:.1f})')
    axes[0, 1].fill_between(thresholds, avg_generations - std_generations, avg_generations + std_generations, alpha=0.2, color='#F18F01')
    for i, (x, y) in enumerate(zip(thresholds, avg_generations)):
        axes[0, 1].text(x, y + std_generations[i] + 10, f'{int(y)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0, 1].set_xlabel('Tournament Threshold', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average Number of Generations', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Generations to Convergence vs Threshold', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0.48, 1.02)
    axes[0, 1].legend(loc='upper left', fontsize=10)

    axes[1, 0].errorbar(thresholds, avg_execution_time, yerr=std_execution_time,
                        marker='s', linewidth=2.5, markersize=8, capsize=5, capthick=2,
                        color='#9B59B6', label='Stochastic Tournament', ecolor='#9B59B6', alpha=0.7)
    axes[1, 0].axhline(y=elite_time, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite ({elite_time:.2f}s)')
    axes[1, 0].axhline(y=tournament_det_time, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det ({tournament_det_time:.2f}s)')
    axes[1, 0].fill_between(thresholds, avg_execution_time - std_execution_time, avg_execution_time + std_execution_time, alpha=0.2, color='#9B59B6')
    for i, (x, y) in enumerate(zip(thresholds, avg_execution_time)):
        axes[1, 0].text(x, y + std_execution_time[i] + 1, f'{y:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1, 0].set_xlabel('Tournament Threshold', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Total Average Execution Time (seconds)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Execution Time vs Threshold', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0.48, 1.02)
    axes[1, 0].legend(loc='upper left', fontsize=10)

    axes[1, 1].errorbar(thresholds, avg_worst_fitness, yerr=std_worst_fitness,
                        marker='D', linewidth=2.5, markersize=8, capsize=5, capthick=2,
                        color='#A23B72', label='Stochastic Tournament', ecolor='#A23B72', alpha=0.7)
    axes[1, 1].axhline(y=elite_fitness, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite ({elite_fitness:.4f})')
    axes[1, 1].axhline(y=tournament_det_fitness, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det ({tournament_det_fitness:.4f})')
    axes[1, 1].fill_between(thresholds, avg_worst_fitness - std_worst_fitness, avg_worst_fitness + std_worst_fitness, alpha=0.2, color='#A23B72')
    for i, (x, y) in enumerate(zip(thresholds, avg_worst_fitness)):
        axes[1, 1].text(x, y + std_worst_fitness[i] + 0.02, f'{y:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1, 1].set_xlabel('Tournament Threshold', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Average Worst Final Fitness', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Worst Final Fitness vs Threshold', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0.48, 1.02)
    axes[1, 1].legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, "threshold_analysis_complete.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    for metric, color, marker, ylabel, title, fname, std_arr, avg_arr in [
        ("fitness", '#2E86AB', 'o', 'Average Best Fitness', 'Final Best Fitness Score vs Stochastic Tournament Threshold',
         "fitness_vs_threshold.png", std_best_fitness, avg_best_fitness),
        ("generations", '#F18F01', '^', 'Average Number of Generations', 'Convergence Speed vs Stochastic Tournament Threshold',
         "generations_vs_threshold.png", std_generations, avg_generations),
        ("time", '#9B59B6', 's', 'Total Average Execution Time (seconds)', 'Execution Time vs Stochastic Tournament Threshold',
         "execution_time_vs_threshold.png", std_execution_time, avg_execution_time),
    ]:
        fig_m, ax_m = plt.subplots(figsize=(12, 7))
        ax_m.errorbar(thresholds, avg_arr, yerr=std_arr,
                      marker=marker, linewidth=2.5, markersize=10, capsize=6, capthick=2.5,
                      color=color, label='Stochastic Tournament', ecolor=color, alpha=0.7)
        if metric == "fitness":
            ax_m.axhline(y=elite_fitness, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite (μ={elite_fitness:.4f})')
            ax_m.axhline(y=tournament_det_fitness, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det (μ={tournament_det_fitness:.4f})')
            for i, (x, y) in enumerate(zip(thresholds, avg_arr)):
                ax_m.text(x, y + std_arr[i] + 0.02, f'{y:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        elif metric == "generations":
            ax_m.axhline(y=elite_generations, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite (μ={elite_generations:.1f})')
            ax_m.axhline(y=tournament_det_generations, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det (μ={tournament_det_generations:.1f})')
            for i, (x, y) in enumerate(zip(thresholds, avg_arr)):
                ax_m.text(x, y + std_arr[i] + 10, f'{int(y)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax_m.axhline(y=elite_time, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite (μ={elite_time:.2f}s)')
            ax_m.axhline(y=tournament_det_time, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det (μ={tournament_det_time:.2f}s)')
            for i, (x, y) in enumerate(zip(thresholds, avg_arr)):
                ax_m.text(x, y + std_arr[i] + 1, f'{y:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax_m.fill_between(thresholds, avg_arr - std_arr, avg_arr + std_arr, alpha=0.2, color=color)
        ax_m.set_xlabel('Tournament Threshold', fontsize=13, fontweight='bold')
        ax_m.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax_m.set_title(title, fontsize=14, fontweight='bold')
        ax_m.grid(True, alpha=0.3)
        ax_m.set_xlim(0.48, 1.02)
        ax_m.legend(loc='lower right' if metric == "fitness" else 'upper left', fontsize=11)
        fig_m.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {output_dir}/")
    print("\nPlots generated:")
    print("  - threshold_analysis_complete.png (4 subplots)")
    print("  - fitness_vs_threshold.png")
    print("  - generations_vs_threshold.png")
    print("  - execution_time_vs_threshold.png")


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
        description="Analyze the effect of stochastic tournament threshold on GA performance."
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file (values will be used as defaults, "
             "overridden by CLI arguments)",
    )
    
    parser.add_argument("--input-image", type=str, help="Path to the input image")
    parser.add_argument("--num-runs", type=int, help="Number of runs per threshold (default: 5)")
    parser.add_argument("--num-thresholds", type=int, help="Number of thresholds to test (default: 10)")
    parser.add_argument("--triangles", type=int, help="Number of triangles (default: 200)")
    parser.add_argument("--population-size", type=int, help="Population size (default: 100)")
    parser.add_argument("--generations", type=int, help="Max generations (default: 1000)")
    parser.add_argument("--k", type=int, help="Offspring size (default: 40)")
    parser.add_argument("--crossover", type=str, choices=["annular", "one_point", "two_point", "uniform"], help="Crossover operator (default: annular)")
    parser.add_argument("--fitness", type=str, choices=["mae", "mse", "rmse", "mse5", "rmse5"], help="Fitness function (default: rmse5)")
    parser.add_argument("--survival-strategy", type=str, choices=["additive", "exclusive"], help="Survival strategy (default: exclusive)")
    parser.add_argument("--mutation", type=str, choices=["gen", "multigen_limited", "multigen_uniform", "complete"], help="Mutation operator (default: multigen_limited)")
    parser.add_argument("--mutation-rate", type=float, help="Mutation rate (default: 0.22)")
    parser.add_argument("--mutation-strength", type=float, help="Mutation strength (default: 0.28)")
    parser.add_argument("--convergence-window", type=int, help="Stop when no improvement over this many generations (default: disabled)")
    parser.add_argument("--convergence-delta", type=float, help="Minimum improvement to count as progress (default: 1e-4)")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: output/threshold_analysis)")
    parser.add_argument("--workers", type=int, help="Number of parallel workers (default: 1)")
    parser.add_argument("--verbose", action="store_true", help="Print generation-by-generation progress")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility (generates unique seeds per run)")

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
    
    run_threshold_analysis(
        input_image_path=input_image,
        num_runs=args.num_runs or config.get("num_runs", 5),
        num_thresholds=args.num_thresholds or config.get("num_thresholds", 10),
        triangles=args.triangles or config.get("triangles", 200),
        population_size=args.population_size or config.get("population_size", 100),
        generations=args.generations or config.get("generations", 1000),
        k=args.k or config.get("k", 40),
        crossover=args.crossover or config.get("crossover", "annular"),
        fitness=args.fitness or config.get("fitness", "rmse5"),
        survival_strategy=args.survival_strategy or config.get("survival_strategy", "exclusive"),
        mutation=args.mutation or config.get("mutation", "multigen_limited"),
        mutation_rate=args.mutation_rate or config.get("mutation_rate", 0.22),
        mutation_strength=args.mutation_strength or config.get("mutation_strength", 0.28),
        convergence_window=args.convergence_window or config.get("convergence_window"),
        convergence_delta=args.convergence_delta or config.get("convergence_delta", 1e-4),
        output_dir=args.output_dir or config.get("output_dir", "output/threshold_analysis"),
        workers=args.workers or config.get("workers", 1),
        verbose=args.verbose or config.get("verbose", False),
        seed=args.seed or config.get("seed"),
    )


if __name__ == "__main__":
    main()
