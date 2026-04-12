"""
Overlay threshold analysis results from multiple input flags.

Plots fitness curves from different threshold analyses on the same figure,
with each curve labeled by the input file name. This helps identify if
threshold behavior is consistent across different flag complexities.

Usage:
  python scripts/plot_threshold_overlay.py \
    --results output/threshold_analysis/results.json \
    --results output/threshold_analysis_france_v2/results.json \
    --results output/threshold_analysis_pattern/results.json \
    --input-names "France" "France v2" "Pattern" \
    --output plots/threshold_overlay_comparison.png

  # Or auto-generate names from result file paths (parent directory names)
  python scripts/plot_threshold_overlay.py \
    --results output/threshold_analysis/results.json \
    --results output/threshold_analysis_1000gens/results.json \
    --auto-names dir
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError as e:
    raise SystemExit(
        "matplotlib and numpy are required. Install with: pip install matplotlib numpy"
    ) from e


# Visual style aligned with TPE2 scripts (from plot_experiment_comparisons.py)
STYLE = {
    "figure_bg": "#fff5ec",
    "axes_bg": "#fff5ec",
    "text_title": "#343434",
    "text_axis": "#343434",
    "grid": "#e8dcd0",
    "grid_minor": "#d4c8bc",
    "stats_text": "#555555",
}

PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "#343434",
    "axes.labelcolor": "#343434",
    "axes.linewidth": 0.8,
    "xtick.color": "#343434",
    "ytick.color": "#343434",
}

# Color palette for multiple curves
CURVE_COLORS = [
    "#4a90d9",  # blue
    "#e67e22",  # orange
    "#27ae60",  # green
    "#c0392b",  # red
    "#8e44ad",  # purple
    "#16a085",  # teal
    "#f39c12",  # gold
    "#2980b9",  # dark blue
    "#d35400",  # dark orange
    "#7f8c8d",  # gray
]

SAVE_PAD_INCHES = 0.2


def load_threshold_results(results_path: str) -> dict:
    """Load results from a threshold analysis JSON file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from {results_path}: {e}")
        sys.exit(1)


def extract_label_from_path(path: str, label_mode: str) -> str:
    """Extract a human-readable label from the results file path."""
    p = Path(path)
    
    if label_mode == "dir":
        # Use parent directory name
        return p.parent.name
    elif label_mode == "file":
        # Use filename without extension
        return p.stem
    else:
        # Full relative path
        return str(p)


def plot_threshold_overlay(
    results_paths: list[str],
    input_names: list[str] | None = None,
    auto_names: str | None = None,
    output_path: str | None = None,
    metric: str = "avg_best_fitness",
    include_std: bool = True,
) -> None:
    """
    Create an overlay plot of threshold analysis results.
    
    Args:
        results_paths: List of paths to results.json files
        input_names: List of custom names for each curve (if not provided, uses auto_names)
        auto_names: How to auto-generate names if input_names not provided:
                   "dir" = parent directory, "file" = filename, None = full path
        output_path: Path to save the plot (if None, displays instead)
        metric: Which metric to plot (avg_best_fitness, avg_worst_fitness, etc.)
        include_std: Whether to shade standard deviation bands
    """
    
    # Apply matplotlib style
    plt.rcParams.update(PLOT_RC)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])
    
    # Prepare names
    if input_names is None:
        if auto_names is None:
            auto_names = "dir"
        input_names = [
            extract_label_from_path(p, auto_names)
            for p in results_paths
        ]
    elif len(input_names) != len(results_paths):
        print(f"Error: number of names ({len(input_names)}) doesn't match "
              f"number of results files ({len(results_paths)})")
        sys.exit(1)
    
    # Determine the corresponding std metric
    std_metric = metric.replace("avg_", "std_")
    
    # Load and plot each results file
    for idx, (results_path, name) in enumerate(zip(results_paths, input_names)):
        results = load_threshold_results(results_path)
        
        # Validate that the metric exists
        if metric not in results:
            print(f"Error: Metric '{metric}' not found in {results_path}")
            print(f"Available metrics: {list(results.keys())}")
            sys.exit(1)
        
        thresholds = np.array(results["thresholds"])
        fitness_values = np.array(results[metric])
        
        # Get color for this curve
        color = CURVE_COLORS[idx % len(CURVE_COLORS)]
        
        # Plot main curve
        ax.plot(
            thresholds,
            fitness_values,
            marker="o",
            linewidth=2,
            markersize=6,
            label=name,
            color=color,
            zorder=3,
        )
        
        # Add standard deviation band if available
        if include_std and std_metric in results:
            std_values = np.array(results[std_metric])
            ax.fill_between(
                thresholds,
                fitness_values - std_values,
                fitness_values + std_values,
                alpha=0.15,
                color=color,
                zorder=1,
            )
    
    # Formatting
    ax.set_xlabel("Tournament Threshold", color=STYLE["text_axis"])
    ax.set_ylabel("Fitness Score", color=STYLE["text_axis"])
    
    # Use metric name for title
    metric_display = metric.replace("_", " ").title()
    ax.set_title(f"Threshold Analysis Overlay: {metric_display}", 
                 color=STYLE["text_title"], pad=15)
    
    # Grid
    ax.grid(True, color=STYLE["grid"], linestyle="-", linewidth=0.5, zorder=0)
    ax.grid(True, which="minor", color=STYLE["grid_minor"], 
            linestyle=":", linewidth=0.3, zorder=0)
    ax.minorticks_on()
    
    # Legend
    ax.legend(
        loc="best",
        framealpha=0.95,
        fancybox=True,
        shadow=False,
        frameon=True,
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=SAVE_PAD_INCHES,
            facecolor=STYLE["figure_bg"],
        )
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--results",
        type=str,
        action="append",
        dest="results",
        required=True,
        help="Path to results.json file from threshold_analyzer (repeatable)",
    )
    
    parser.add_argument(
        "--input-names",
        type=str,
        nargs="+",
        help="Custom names for each curve (space-separated)",
    )
    
    parser.add_argument(
        "--auto-names",
        type=str,
        choices=["dir", "file", "path"],
        default="dir",
        help="How to auto-generate names if --input-names not provided "
             "(dir=parent dir, file=filename, path=full path) [default: dir]",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the plot (for example: plots/overlay.png). "
             "If not provided, displays plot instead.",
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="avg_best_fitness",
        help="Which metric to plot [default: avg_best_fitness]",
    )
    
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Don't show standard deviation bands",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if len(args.results) < 2:
        print("Error: At least 2 result files are required for overlay")
        sys.exit(1)
    
    plot_threshold_overlay(
        results_paths=args.results,
        input_names=args.input_names,
        auto_names=args.auto_names if not args.input_names else None,
        output_path=args.output,
        metric=args.metric,
        include_std=not args.no_std,
    )


if __name__ == "__main__":
    main()
