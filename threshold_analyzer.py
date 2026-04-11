"""
Analyzes the effect of stochastic tournament threshold on GA performance.

Runs the GA multiple times with uniformly-spaced thresholds from 0.5 to 1.0,
using stochastic tournament selection and no_improvement stop condition.

Plots:
1. Final best fitness score vs threshold
2. Worst final fitness score vs threshold
3. Number of generations vs threshold
4. Average execution time vs threshold
5. Comparison with elite and tournament deterministic selections

Usage:
  python threshold_analyzer.py --input-image input/Flag_of_France.svg.png \
                               --num-runs 5 \
                               --num-thresholds 10 \
                               --convergence-window 20
"""

import argparse
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
    output_dir: str = "output/threshold_analysis",
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

    with Image.open(BytesIO(image_bytes)) as img:
        source_image = img.convert("RGBA").copy()

    # Generate uniformly spaced thresholds
    thresholds = np.linspace(0.5, 1.0, num_thresholds)
    
    # Results storage
    results = {
        "thresholds": [],
        "best_fitness_scores": [],      # list of lists (one per threshold)
        "worst_fitness_scores": [],     # list of lists
        "generations_run": [],          # list of lists
        "execution_times": [],          # list of lists
        "avg_best_fitness": [],
        "std_best_fitness": [],
        "avg_worst_fitness": [],
        "std_worst_fitness": [],
        "avg_generations": [],
        "std_generations": [],
        "avg_execution_time": [],
        "std_execution_time": [],
    }
    
    # Baseline results (elite and tournament_det)
    baseline_results = {
        "elite": {"best_fitness": [], "generations": [], "execution_time": []},
        "tournament_det": {"best_fitness": [], "generations": [], "execution_time": []},
    }
    
    total_experiments = num_thresholds * num_runs + 2 * num_runs  # +2*num_runs for baselines
    experiment_count = 0
    
    print(f"Running {total_experiments} experiments ({num_runs} runs per threshold, +{num_runs} for each baseline)...\n")
    
    # Run baseline experiments (elite and tournament_deterministic)
    print("=" * 70)
    print("BASELINE EXPERIMENTS")
    print("=" * 70)
    
    for baseline_name in ["elite", "tournament_det"]:
        print(f"\nRunning {baseline_name} baseline ({num_runs} runs)...")
        for run in range(num_runs):
            experiment_count += 1
            print(f"[{experiment_count}/{total_experiments}] Baseline={baseline_name}, Run={run+1}/{num_runs}")
            
            try:
                start_time = time.time()
                result = run_genetic_algorithm(
                    source_image=source_image,
                    num_triangles=triangles,
                    population_size=population_size,
                    generations=generations,
                    k=k,
                    selector=build_selector(
                        baseline_name,
                        temperature=50.0,
                        temperature_min=1.0,
                        temperature_decay=-0.005,
                        tournament_threshold=0.5,  # Not used for elite/tournament_det
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
                        convergence_window=None,
                        convergence_delta=None,
                        time_limit_secs=None,
                    ),
                )
                elapsed_time = time.time() - start_time
                
                baseline_results[baseline_name]["best_fitness"].append(result.best_fitness)
                baseline_results[baseline_name]["generations"].append(result.generations_run)
                baseline_results[baseline_name]["execution_time"].append(elapsed_time)
                
                print(f"    best_fitness={result.best_fitness:.6f}  generations={result.generations_run}  time={elapsed_time:.2f}s")
                
            except Exception as e:
                print(f"    Error in run {run+1}: {e}")
                continue
    
    # Run stochastic tournament experiments with varying thresholds
    print("\n" + "=" * 70)
    print("STOCHASTIC TOURNAMENT EXPERIMENTS")
    print("=" * 70 + "\n")
    
    threshold_idx = 0
    for threshold in thresholds:
        threshold_idx += 1
        threshold_results = {
            "best_fitness": [],
            "worst_fitness": [],
            "generations": [],
            "execution_time": [],
        }
        
        print(f"Threshold {threshold_idx}/{num_thresholds}: {threshold:.3f}")
        
        for run in range(num_runs):
            experiment_count += 1
            print(f"[{experiment_count}/{total_experiments}] Threshold={threshold:.3f}, Run={run+1}/{num_runs}")
            
            try:
                start_time = time.time()
                result = run_genetic_algorithm(
                    source_image=source_image,
                    num_triangles=triangles,
                    population_size=population_size,
                    generations=generations,
                    k=k,
                    selector=build_selector(
                        "tournament_stoch",
                        temperature=50.0,
                        temperature_min=1.0,
                        temperature_decay=-0.005,
                        tournament_threshold=threshold,
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
                elapsed_time = time.time() - start_time
                
                threshold_results["best_fitness"].append(result.best_fitness)
                threshold_results["worst_fitness"].append(result.best_fitness)
                threshold_results["generations"].append(result.generations_run)
                threshold_results["execution_time"].append(elapsed_time)
                
                # Save final image for first run of each threshold
                if run == 0:
                    image_filename = f"threshold_{threshold:.3f}_final.png"
                    image_path = os.path.join(images_dir, image_filename)
                    save_image(result.image_bytes, image_path)
                
                print(f"    best_fitness={result.best_fitness:.6f}  generations={result.generations_run}  time={elapsed_time:.2f}s")
                
            except Exception as e:
                print(f"    Error in run {run+1}: {e}")
                continue
        
        if threshold_results["best_fitness"]:  # Only if we got results
            results["thresholds"].append(threshold)
            results["best_fitness_scores"].append(threshold_results["best_fitness"])
            results["worst_fitness_scores"].append(threshold_results["worst_fitness"])
            results["generations_run"].append(threshold_results["generations"])
            results["execution_times"].append(threshold_results["execution_time"])
            results["avg_best_fitness"].append(np.mean(threshold_results["best_fitness"]))
            results["std_best_fitness"].append(np.std(threshold_results["best_fitness"]))
            results["avg_worst_fitness"].append(np.mean(threshold_results["worst_fitness"]))
            results["std_worst_fitness"].append(np.std(threshold_results["worst_fitness"]))
            results["avg_generations"].append(np.mean(threshold_results["generations"]))
            results["std_generations"].append(np.std(threshold_results["generations"]))
            results["avg_execution_time"].append(np.mean(threshold_results["execution_time"]))
            results["std_execution_time"].append(np.std(threshold_results["execution_time"]))
        
        print()

    # Save results to JSON
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
            "baseline_elite_fitness": np.mean(baseline_results["elite"]["best_fitness"]),
            "baseline_elite_generations": np.mean(baseline_results["elite"]["generations"]),
            "baseline_elite_time": np.mean(baseline_results["elite"]["execution_time"]),
            "baseline_tournament_det_fitness": np.mean(baseline_results["tournament_det"]["best_fitness"]),
            "baseline_tournament_det_generations": np.mean(baseline_results["tournament_det"]["generations"]),
            "baseline_tournament_det_time": np.mean(baseline_results["tournament_det"]["execution_time"]),
        }, f, indent=2)
    print(f"Results saved to {results_json_path}\n")
    print(f"Final images saved to {images_dir}\n")

    # Create plots
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
    
    # Baseline values
    elite_fitness = np.mean(baseline_results["elite"]["best_fitness"])
    elite_generations = np.mean(baseline_results["elite"]["generations"])
    elite_time = np.mean(baseline_results["elite"]["execution_time"])
    
    tournament_det_fitness = np.mean(baseline_results["tournament_det"]["best_fitness"])
    tournament_det_generations = np.mean(baseline_results["tournament_det"]["generations"])
    tournament_det_time = np.mean(baseline_results["tournament_det"]["execution_time"])
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Final best fitness score vs threshold
    axes[0, 0].errorbar(thresholds, avg_best_fitness, yerr=std_best_fitness, 
                        marker='o', linewidth=2.5, markersize=8, capsize=5, capthick=2, 
                        color='#2E86AB', label='Stochastic Tournament', ecolor='#2E86AB', alpha=0.7)
    axes[0, 0].axhline(y=elite_fitness, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite ({elite_fitness:.4f})')
    axes[0, 0].axhline(y=tournament_det_fitness, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det ({tournament_det_fitness:.4f})')
    axes[0, 0].fill_between(thresholds, avg_best_fitness - std_best_fitness, avg_best_fitness + std_best_fitness, 
                            alpha=0.2, color='#2E86AB')
    
    # Add labels to stochastic tournament points
    for i, (x, y) in enumerate(zip(thresholds, avg_best_fitness)):
        axes[0, 0].text(x, y + std_best_fitness[i] + 0.02, f'{y:.3f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    axes[0, 0].set_xlabel('Tournament Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Average Best Fitness', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Final Best Fitness vs Threshold', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0.48, 1.02)
    axes[0, 0].legend(loc='lower right', fontsize=10)
    
    # Plot 2: Number of generations vs threshold
    axes[0, 1].errorbar(thresholds, avg_generations, yerr=std_generations, 
                        marker='^', linewidth=2.5, markersize=8, capsize=5, capthick=2, 
                        color='#F18F01', label='Stochastic Tournament', ecolor='#F18F01', alpha=0.7)
    axes[0, 1].axhline(y=elite_generations, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite ({elite_generations:.1f})')
    axes[0, 1].axhline(y=tournament_det_generations, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det ({tournament_det_generations:.1f})')
    axes[0, 1].fill_between(thresholds, avg_generations - std_generations, avg_generations + std_generations, 
                            alpha=0.2, color='#F18F01')
    
    # Add labels to stochastic tournament points
    for i, (x, y) in enumerate(zip(thresholds, avg_generations)):
        axes[0, 1].text(x, y + std_generations[i] + 10, f'{int(y)}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    axes[0, 1].set_xlabel('Tournament Threshold', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average Number of Generations', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Generations to Convergence vs Threshold', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0.48, 1.02)
    axes[0, 1].legend(loc='upper left', fontsize=10)
    
    # Plot 3: Execution time vs threshold
    axes[1, 0].errorbar(thresholds, avg_execution_time, yerr=std_execution_time, 
                        marker='s', linewidth=2.5, markersize=8, capsize=5, capthick=2, 
                        color='#9B59B6', label='Stochastic Tournament', ecolor='#9B59B6', alpha=0.7)
    axes[1, 0].axhline(y=elite_time, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite ({elite_time:.2f}s)')
    axes[1, 0].axhline(y=tournament_det_time, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det ({tournament_det_time:.2f}s)')
    axes[1, 0].fill_between(thresholds, avg_execution_time - std_execution_time, avg_execution_time + std_execution_time, 
                            alpha=0.2, color='#9B59B6')
    
    # Add labels to stochastic tournament points
    for i, (x, y) in enumerate(zip(thresholds, avg_execution_time)):
        axes[1, 0].text(x, y + std_execution_time[i] + 1, f'{y:.1f}s', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    axes[1, 0].set_xlabel('Tournament Threshold', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Total Average Execution Time (seconds)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Execution Time vs Threshold', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0.48, 1.02)
    axes[1, 0].legend(loc='upper left', fontsize=10)
    
    # Plot 4: Worst final fitness vs threshold
    axes[1, 1].errorbar(thresholds, avg_worst_fitness, yerr=std_worst_fitness, 
                        marker='D', linewidth=2.5, markersize=8, capsize=5, capthick=2, 
                        color='#A23B72', label='Stochastic Tournament', ecolor='#A23B72', alpha=0.7)
    axes[1, 1].axhline(y=elite_fitness, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite ({elite_fitness:.4f})')
    axes[1, 1].axhline(y=tournament_det_fitness, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det ({tournament_det_fitness:.4f})')
    axes[1, 1].fill_between(thresholds, avg_worst_fitness - std_worst_fitness, avg_worst_fitness + std_worst_fitness, 
                            alpha=0.2, color='#A23B72')
    
    # Add labels to stochastic tournament points
    for i, (x, y) in enumerate(zip(thresholds, avg_worst_fitness)):
        axes[1, 1].text(x, y + std_worst_fitness[i] + 0.02, f'{y:.3f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
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
    print(f"Complete plot saved to {plot_path}")
    plt.close()
    
    # Create individual high-quality plots
    # Plot fitness only
    fig_fitness, ax_fitness = plt.subplots(figsize=(12, 7))
    ax_fitness.errorbar(thresholds, avg_best_fitness, yerr=std_best_fitness, 
                       marker='o', linewidth=2.5, markersize=10, capsize=6, capthick=2.5, 
                       color='#2E86AB', label='Stochastic Tournament', ecolor='#2E86AB', alpha=0.7)
    ax_fitness.axhline(y=elite_fitness, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite (μ={elite_fitness:.4f})')
    ax_fitness.axhline(y=tournament_det_fitness, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det (μ={tournament_det_fitness:.4f})')
    ax_fitness.fill_between(thresholds, avg_best_fitness - std_best_fitness, avg_best_fitness + std_best_fitness, 
                           alpha=0.2, color='#2E86AB')
    
    # Add labels
    for i, (x, y) in enumerate(zip(thresholds, avg_best_fitness)):
        ax_fitness.text(x, y + std_best_fitness[i] + 0.02, f'{y:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_fitness.set_xlabel('Tournament Threshold', fontsize=13, fontweight='bold')
    ax_fitness.set_ylabel('Average Best Fitness', fontsize=13, fontweight='bold')
    ax_fitness.set_title('Final Best Fitness Score vs Stochastic Tournament Threshold', fontsize=14, fontweight='bold')
    ax_fitness.grid(True, alpha=0.3)
    ax_fitness.set_xlim(0.48, 1.02)
    ax_fitness.legend(loc='lower right', fontsize=11)
    fig_fitness.savefig(os.path.join(output_dir, "fitness_vs_threshold.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot generations only
    fig_gens, ax_gens = plt.subplots(figsize=(12, 7))
    ax_gens.errorbar(thresholds, avg_generations, yerr=std_generations, 
                    marker='^', linewidth=2.5, markersize=10, capsize=6, capthick=2.5, 
                    color='#F18F01', label='Stochastic Tournament', ecolor='#F18F01', alpha=0.7)
    ax_gens.axhline(y=elite_generations, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite (μ={elite_generations:.1f})')
    ax_gens.axhline(y=tournament_det_generations, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det (μ={tournament_det_generations:.1f})')
    ax_gens.fill_between(thresholds, avg_generations - std_generations, avg_generations + std_generations, 
                        alpha=0.2, color='#F18F01')
    
    # Add labels
    for i, (x, y) in enumerate(zip(thresholds, avg_generations)):
        ax_gens.text(x, y + std_generations[i] + 10, f'{int(y)}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_gens.set_xlabel('Tournament Threshold', fontsize=13, fontweight='bold')
    ax_gens.set_ylabel('Average Number of Generations', fontsize=13, fontweight='bold')
    ax_gens.set_title('Convergence Speed vs Stochastic Tournament Threshold', fontsize=14, fontweight='bold')
    ax_gens.grid(True, alpha=0.3)
    ax_gens.set_xlim(0.48, 1.02)
    ax_gens.legend(loc='upper left', fontsize=11)
    fig_gens.savefig(os.path.join(output_dir, "generations_vs_threshold.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot execution time only
    fig_time, ax_time = plt.subplots(figsize=(12, 7))
    ax_time.errorbar(thresholds, avg_execution_time, yerr=std_execution_time, 
                    marker='s', linewidth=2.5, markersize=10, capsize=6, capthick=2.5, 
                    color='#9B59B6', label='Stochastic Tournament', ecolor='#9B59B6', alpha=0.7)
    ax_time.axhline(y=elite_time, color='#06A77D', linestyle='--', linewidth=2.5, label=f'Elite (μ={elite_time:.2f}s)')
    ax_time.axhline(y=tournament_det_time, color='#D62828', linestyle='--', linewidth=2.5, label=f'Tournament Det (μ={tournament_det_time:.2f}s)')
    ax_time.fill_between(thresholds, avg_execution_time - std_execution_time, avg_execution_time + std_execution_time, 
                        alpha=0.2, color='#9B59B6')
    
    # Add labels
    for i, (x, y) in enumerate(zip(thresholds, avg_execution_time)):
        ax_time.text(x, y + std_execution_time[i] + 1, f'{y:.1f}s', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_time.set_xlabel('Tournament Threshold', fontsize=13, fontweight='bold')
    ax_time.set_ylabel('Total Average Execution Time (seconds)', fontsize=13, fontweight='bold')
    ax_time.set_title('Execution Time vs Stochastic Tournament Threshold', fontsize=14, fontweight='bold')
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlim(0.48, 1.02)
    ax_time.legend(loc='upper left', fontsize=11)
    fig_time.savefig(os.path.join(output_dir, "execution_time_vs_threshold.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual plots saved to {output_dir}/")
    print("\nPlots generated:")
    print("  - threshold_analysis_complete.png (4 subplots)")
    print("  - fitness_vs_threshold.png")
    print("  - generations_vs_threshold.png")
    print("  - execution_time_vs_threshold.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the effect of stochastic tournament threshold on GA performance."
    )
    
    parser.add_argument("--input-image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs per threshold (default: 5)")
    parser.add_argument("--num-thresholds", type=int, default=10, help="Number of thresholds to test (default: 10)")
    parser.add_argument("--triangles", type=int, default=200, help="Number of triangles (default: 200)")
    parser.add_argument("--population-size", type=int, default=100, help="Population size (default: 100)")
    parser.add_argument("--generations", type=int, default=1000, help="Max generations (default: 1000)")
    parser.add_argument("--k", type=int, default=40, help="Offspring size (default: 40)")
    parser.add_argument("--crossover", type=str, default="annular", choices=["annular", "one_point", "two_point", "uniform"], help="Crossover operator (default: annular)")
    parser.add_argument("--fitness", type=str, default="rmse5", choices=["mae", "mse", "rmse", "mse5", "rmse5"], help="Fitness function (default: rmse5)")
    parser.add_argument("--survival-strategy", type=str, default="exclusive", choices=["additive", "exclusive"], help="Survival strategy (default: exclusive)")
    parser.add_argument("--mutation", type=str, default="multigen_limited", choices=["gen", "multigen_limited", "multigen_uniform", "complete"], help="Mutation operator (default: multigen_limited)")
    parser.add_argument("--mutation-rate", type=float, default=0.22, help="Mutation rate (default: 0.1)")
    parser.add_argument("--mutation-strength", type=float, default=0.28, help="Mutation strength (default: 0.3)")
    parser.add_argument("--output-dir", type=str, default="output/threshold_analysis", help="Output directory (default: output/threshold_analysis)")
    
    args = parser.parse_args()
    run_threshold_analysis(
        input_image_path=args.input_image,
        num_runs=args.num_runs,
        num_thresholds=args.num_thresholds,
        triangles=args.triangles,
        population_size=args.population_size,
        generations=args.generations,
        k=args.k,
        crossover=args.crossover,
        fitness=args.fitness,
        survival_strategy=args.survival_strategy,
        mutation=args.mutation,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
