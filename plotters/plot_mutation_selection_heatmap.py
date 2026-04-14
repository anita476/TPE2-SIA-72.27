"""
Heatmap visualization of mutation × selection grid experiment.

Plots fitness performance across all mutation-selection combinations,
with standard deviation annotations. Helps identify synergistic pairs
and pressure-mutation pairings.

Usage:
  python scripts/plot_mutation_selection_heatmap.py \
    --results output/mutation_selection_grid/results.json \
    --output plots/mutation_selection_heatmap.png
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
except ImportError:
    raise SystemExit(
        "matplotlib and numpy are required. Install with: pip install matplotlib numpy"
    )


# Visual style aligned with TPE2 scripts
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
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#343434",
    "axes.labelcolor": "#343434",
    "axes.linewidth": 0.8,
    "xtick.color": "#343434",
    "ytick.color": "#343434",
}

SAVE_PAD_INCHES = 0.2


def load_results(results_path: str) -> dict:
    """Load grid experiment results."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON: {e}")
        sys.exit(1)


def plot_heatmap(
    results_path: str,
    output_path: str | None = None,
    metric: str = "avg",
    include_std: bool = True,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Create heatmap of mutation × selection performance.
    
    Args:
        results_path: Path to results.json from grid experiment
        output_path: Where to save plot (if None, displays)
        metric: "avg" or "std"
        include_std: Whether to annotate cells with std values
        cmap: Colormap name (RdYlGn = red-yellow-green)
        vmin, vmax: Color scale limits
    """
    
    # Load results
    results = load_results(results_path)
    
    mutations = results["mutations"]
    selections = results["selections"]
    
    # Extract data
    if metric == "avg":
        data_dict = results["statistics"]["avg_best_fitness"]
        std_dict = results["statistics"]["std_best_fitness"]
    else:
        std_dict = results["statistics"]["std_best_fitness"]
        data_dict = std_dict
    
    # Build matrix
    data_matrix = np.zeros((len(selections), len(mutations)))
    std_matrix = np.zeros((len(selections), len(mutations)))
    
    for i, selection in enumerate(selections):
        for j, mutation in enumerate(mutations):
            if selection in data_dict and mutation in data_dict[selection]:
                data_matrix[i, j] = data_dict[selection][mutation]
            if selection in std_dict and mutation in std_dict[selection]:
                std_matrix[i, j] = std_dict[selection][mutation]
    
    # Apply matplotlib style
    plt.rcParams.update(PLOT_RC)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])
    
    # Set color scale limits
    if vmin is None:
        vmin = data_matrix.min()
    if vmax is None:
        vmax = data_matrix.max()
    
    # Plot heatmap
    im = ax.imshow(
        data_matrix,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(mutations)))
    ax.set_yticks(np.arange(len(selections)))
    ax.set_xticklabels(mutations, rotation=45, ha="right")
    ax.set_yticklabels(selections)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Average Best Fitness", color=STYLE["text_axis"])
    cbar.ax.tick_params(colors=STYLE["text_axis"])
    
    # Annotate cells with fitness values
    for i in range(len(selections)):
        for j in range(len(mutations)):
            text_color = "white" if (data_matrix[i, j] - vmin) / (vmax - vmin) < 0.5 else "black"
            
            # Main value
            text = f"{data_matrix[i, j]:.4f}"
            if include_std:
                text += f"\n±{std_matrix[i, j]:.4f}"
            
            ax.text(
                j, i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8 if include_std else 9,
                weight="bold",
            )
    
    # Title and labels
    ax.set_title(
        "Mutation × Selection Performance Grid\nAverage Best Fitness (Higher = Better)",
        color=STYLE["text_title"],
        pad=15,
    )
    ax.set_xlabel("Mutation Operator", color=STYLE["text_axis"], labelpad=10)
    ax.set_ylabel("Selection Method", color=STYLE["text_axis"], labelpad=10)
    
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
        print(f"Saved heatmap to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)
    
    # Print text summary
    print("\n" + "="*80)
    print("HEATMAP SUMMARY")
    print("="*80)
    
    print("\nAverage Best Fitness by Selection × Mutation:")
    print(f"\n{'Selection':<20}", end="")
    for mutation in mutations:
        print(f"{mutation:>14s}", end="")
    print("\n" + "-"*80)
    
    for i, selection in enumerate(selections):
        print(f"{selection:<20}", end="")
        for j, mutation in enumerate(mutations):
            val = data_matrix[i, j]
            std = std_matrix[i, j]
            print(f"{val:>7.4f}±{std:<6.4f}", end="")
        print()
    
    # Identify best combinations
    print("\n" + "="*80)
    print("TOP 5 BEST COMBINATIONS")
    print("="*80)
    
    flat_indices = np.argsort(data_matrix.ravel())[::-1][:5]
    for rank, idx in enumerate(flat_indices, 1):
        i, j = np.unravel_index(idx, data_matrix.shape)
        selection = selections[i]
        mutation = mutations[j]
        fitness = data_matrix[i, j]
        std = std_matrix[i, j]
        print(f"{rank}. {selection:20s} × {mutation:20s}: {fitness:.6f} ± {std:.6f}")
    
    # Identify pressure-mutation pairings
    print("\n" + "="*80)
    print("PRESSURE-MUTATION PAIRING ANALYSIS")
    print("="*80)
    
    # High-pressure: elite, tournament_det
    high_pressure = ["elite", "tournament_det"]
    # Low-pressure: roulette, universal
    low_pressure = ["roulette", "universal"]
    # Conservative mutation: gen (single)
    # Aggressive mutation: complete
    conservative = ["gen"]
    aggressive = ["complete"]
    
    print("\nHigh-Pressure Selection (elite, tournament_det):")
    hp_results = {}
    for selection in high_pressure:
        sel_idx = selections.index(selection)
        for mutation in mutations:
            mut_idx = mutations.index(mutation)
            val = data_matrix[sel_idx, mut_idx]
            if mutation not in hp_results:
                hp_results[mutation] = []
            hp_results[mutation].append(val)
    
    for mutation in mutations:
        avg = np.mean(hp_results[mutation])
        is_cons = "conservative" if mutation in conservative else ("aggressive" if mutation in aggressive else "moderate")
        print(f"  {mutation:20s} ({is_cons:12s}): {avg:.6f}")
    
    print("\nLow-Pressure Selection (roulette, universal):")
    lp_results = {}
    for selection in low_pressure:
        sel_idx = selections.index(selection)
        for mutation in mutations:
            mut_idx = mutations.index(mutation)
            val = data_matrix[sel_idx, mut_idx]
            if mutation not in lp_results:
                lp_results[mutation] = []
            lp_results[mutation].append(val)
    
    for mutation in mutations:
        avg = np.mean(lp_results[mutation])
        is_cons = "conservative" if mutation in conservative else ("aggressive" if mutation in aggressive else "moderate")
        print(f"  {mutation:20s} ({is_cons:12s}): {avg:.6f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results.json from mutation_selection_grid_experiment.py",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save heatmap (e.g., plots/heatmap.png). "
             "If not provided, displays instead.",
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        choices=["avg", "std"],
        default="avg",
        help="Plot average or standard deviation [default: avg]",
    )
    
    parser.add_argument(
        "--no-std-annotations",
        action="store_true",
        help="Don't show std values in cells",
    )
    
    parser.add_argument(
        "--cmap",
        type=str,
        default="RdYlGn",
        help="Matplotlib colormap name [default: RdYlGn]",
    )
    
    parser.add_argument(
        "--vmin",
        type=float,
        help="Minimum value for color scale (auto if not provided)",
    )
    
    parser.add_argument(
        "--vmax",
        type=float,
        help="Maximum value for color scale (auto if not provided)",
    )
    
    args = parser.parse_args()
    
    plot_heatmap(
        results_path=args.results,
        output_path=args.output,
        metric=args.metric,
        include_std=not args.no_std_annotations,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
