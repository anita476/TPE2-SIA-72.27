"""
Plot Tournament Size (M) vs Triangle Count Results
 
Reads the summary.json (and optionally per-triangle JSONs) produced by
tournament_m_triangles_experiment.py and generates 4 plots:
 
  1. fitness_heatmap.png       — heatmap: rows=triangles, cols=M, color=avg fitness
  2. fitness_lines.png         — one curve per triangle count, x=M, y=avg fitness ± std
  3. optimal_m_scaling.png     — optimal M as a function of triangle count
  4. fitness_per_triangle/     — individual subplot per triangle count (like original script)
 
Usage:
  python plot_tournament_m_triangles.py \
    --summary output/tournament_m_triangles/summary.json \
    --results-dir output/tournament_m_triangles/ \
    --output plots/tournament_m_triangles/
"""
 
import argparse
import json
import sys
from pathlib import Path
 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
 
 
# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────
 
def load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON: {e}")
        sys.exit(1)
 
 
def load_per_triangle_results(results_dir: str, triangle_counts: list[int]) -> dict[int, dict]:
    """Load individual results_triangles_N.json files if available."""
    per_triangle = {}
    for t in triangle_counts:
        path = Path(results_dir) / f"results_triangles_{t}.json"
        if path.exists():
            per_triangle[t] = load_json(str(path))
        else:
            print(f"  Warning: {path} not found — skipping per-triangle detail plots for {t} triangles")
    return per_triangle
 
 
# ─────────────────────────────────────────────
# Plot 1: Heatmap
# ─────────────────────────────────────────────
 
def plot_heatmap(summary: dict, output_dir: Path) -> None:
    triangle_counts = summary["triangle_counts"]
    tournament_sizes = summary["tournament_sizes"]
    avg_fitness = summary["avg_fitness"]
 
    # Build matrix
    matrix = np.zeros((len(triangle_counts), len(tournament_sizes)))
    for i, t in enumerate(triangle_counts):
        for j, m in enumerate(tournament_sizes):
            matrix[i, j] = avg_fitness.get(str(t), {}).get(str(m), np.nan)
 
    fig, ax = plt.subplots(figsize=(12, 5))
 
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Avg Best Fitness")
 
    # Annotate cells
    for i in range(len(triangle_counts)):
        for j in range(len(tournament_sizes)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if val < matrix.max() * 0.8 else "white")
 
    # Mark optimal M per row with a star
    for i, t in enumerate(triangle_counts):
        opt = summary["optimal_m_per_triangle"].get(str(t), {})
        opt_m = opt.get("optimal_m")
        if opt_m and opt_m in tournament_sizes:
            j = tournament_sizes.index(opt_m)
            ax.plot(j, i, marker="*", markersize=14, color="blue",
                    markeredgecolor="white", markeredgewidth=0.8, zorder=5)
 
    ax.set_xticks(range(len(tournament_sizes)))
    ax.set_xticklabels([str(m) for m in tournament_sizes], fontsize=10)
    ax.set_yticks(range(len(triangle_counts)))
    ax.set_yticklabels([str(t) for t in triangle_counts], fontsize=10)
    ax.set_xlabel("Tournament Size (M)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Triangles", fontsize=12, fontweight="bold")
    ax.set_title("Avg Best Fitness Heatmap  (★ = optimal M per row)", fontsize=13, fontweight="bold")
 
    plt.tight_layout()
    out = output_dir / "fitness_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
 
 
# ─────────────────────────────────────────────
# Plot 2: Fitness lines (one per triangle count)
# ─────────────────────────────────────────────
 
def plot_fitness_lines(summary: dict, per_triangle: dict[int, dict], output_dir: Path) -> None:
    triangle_counts = summary["triangle_counts"]
    tournament_sizes = summary["tournament_sizes"]
 
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(triangle_counts)))
 
    fig, ax = plt.subplots(figsize=(13, 6))
 
    for idx, t in enumerate(triangle_counts):
        avg_vals = []
        std_vals = []
 
        # Prefer per-triangle JSON for std, fall back to summary (no std)
        if t in per_triangle:
            stats = per_triangle[t]["statistics"]
            for m in tournament_sizes:
                avg_vals.append(stats["avg_best_fitness"].get(str(m), np.nan))
                std_vals.append(stats["std_best_fitness"].get(str(m), 0.0))
        else:
            for m in tournament_sizes:
                avg_vals.append(summary["avg_fitness"].get(str(t), {}).get(str(m), np.nan))
                std_vals.append(0.0)
 
        color = colors[idx]
        avg_arr = np.array(avg_vals)
        std_arr = np.array(std_vals)
 
        ax.plot(tournament_sizes, avg_arr, marker="o", markersize=7,
                linewidth=2.2, color=color, label=f"{t} triangles")
        ax.fill_between(tournament_sizes,
                        avg_arr - std_arr,
                        avg_arr + std_arr,
                        alpha=0.12, color=color)
 
        # Mark optimal M
        valid = ~np.isnan(avg_arr)
        if valid.any():
            opt_idx = np.argmax(avg_arr[valid])
            opt_x = np.array(tournament_sizes)[valid][opt_idx]
            opt_y = avg_arr[valid][opt_idx]
            ax.scatter([opt_x], [opt_y], s=160, marker="*",
                       color=color, zorder=5, edgecolors="black", linewidth=0.8)
 
    ax.set_xlabel("Tournament Size (M)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Best Fitness", fontsize=12, fontweight="bold")
    ax.set_title("Fitness vs M — by Triangle Count  (★ = optimal M)", fontsize=13, fontweight="bold")
    ax.set_xticks(tournament_sizes)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10)
 
    plt.tight_layout()
    out = output_dir / "fitness_lines.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
 
 
# ─────────────────────────────────────────────
# Plot 3: Optimal M scaling
# ─────────────────────────────────────────────
 
def plot_optimal_m_scaling(summary: dict, output_dir: Path) -> None:
    triangle_counts = summary["triangle_counts"]
    opt_data = summary["optimal_m_per_triangle"]
 
    opt_ms = [opt_data[str(t)]["optimal_m"] for t in triangle_counts]
    opt_fits = [opt_data[str(t)]["fitness"] for t in triangle_counts]
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
 
    # Left: optimal M vs triangles
    ax1.plot(triangle_counts, opt_ms, marker="o", markersize=9,
             linewidth=2.5, color="#2196F3")
    for x, y in zip(triangle_counts, opt_ms):
        ax1.annotate(f"M={y}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
 
    ax1.set_xlabel("Number of Triangles", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Optimal Tournament Size (M)", fontsize=12, fontweight="bold")
    ax1.set_title("Optimal M vs Triangle Count", fontsize=12, fontweight="bold")
    ax1.set_xticks(triangle_counts)
    ax1.set_yticks(sorted(set(opt_ms)))
    ax1.grid(True, alpha=0.3, linestyle="--")
 
    # Right: best achievable fitness vs triangles
    ax2.plot(triangle_counts, opt_fits, marker="s", markersize=9,
             linewidth=2.5, color="#4CAF50")
    for x, y in zip(triangle_counts, opt_fits):
        ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.7))
 
    ax2.set_xlabel("Number of Triangles", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Best Avg Fitness at Optimal M", fontsize=12, fontweight="bold")
    ax2.set_title("Best Achievable Fitness vs Triangle Count", fontsize=12, fontweight="bold")
    ax2.set_xticks(triangle_counts)
    ax2.grid(True, alpha=0.3, linestyle="--")
 
    plt.suptitle("How Optimal M Scales with Problem Complexity", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_dir / "optimal_m_scaling.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
 
 
# ─────────────────────────────────────────────
# Plot 4: Individual subplots per triangle count
# ─────────────────────────────────────────────
 
def plot_individual_subplots(summary: dict, per_triangle: dict[int, dict], output_dir: Path) -> None:
    triangle_counts = summary["triangle_counts"]
    tournament_sizes = summary["tournament_sizes"]
    n = len(triangle_counts)
 
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()
 
    colors = plt.cm.tab10.colors
 
    for idx, t in enumerate(triangle_counts):
        ax = axes[idx]
        color = colors[idx % len(colors)]
 
        if t in per_triangle:
            stats = per_triangle[t]["statistics"]
            avg_vals = [stats["avg_best_fitness"].get(str(m), np.nan) for m in tournament_sizes]
            std_vals = [stats["std_best_fitness"].get(str(m), 0.0) for m in tournament_sizes]
        else:
            avg_vals = [summary["avg_fitness"].get(str(t), {}).get(str(m), np.nan) for m in tournament_sizes]
            std_vals = [0.0] * len(tournament_sizes)
 
        avg_arr = np.array(avg_vals)
        std_arr = np.array(std_vals)
 
        ax.plot(tournament_sizes, avg_arr, marker="o", markersize=7,
                linewidth=2.2, color=color, label="Mean Fitness")
        ax.fill_between(tournament_sizes,
                        avg_arr - std_arr,
                        avg_arr + std_arr,
                        alpha=0.2, color=color, label="±1 Std Dev")
 
        # Optimal star
        valid = ~np.isnan(avg_arr)
        if valid.any():
            opt_idx = np.argmax(avg_arr[valid])
            opt_x = np.array(tournament_sizes)[valid][opt_idx]
            opt_y = avg_arr[valid][opt_idx]
            ax.scatter([opt_x], [opt_y], s=200, marker="*",
                       color="red", zorder=5, edgecolors="darkred", linewidth=1.2)
            ax.text(opt_x, opt_y, f"  M={opt_x}\n  {opt_y:.4f}",
                    fontsize=8, ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.3))
 
        ax.set_title(f"{t} Triangles", fontsize=11, fontweight="bold")
        ax.set_xlabel("Tournament Size (M)", fontsize=10)
        ax.set_ylabel("Avg Best Fitness", fontsize=10)
        ax.set_xticks(tournament_sizes)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)
 
    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
 
    plt.suptitle("Fitness vs M — Individual Triangle Counts", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "fitness_per_triangle.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
 
 
# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────
 
def print_and_save_summary(summary: dict, output_dir: Path) -> None:
    triangle_counts = summary["triangle_counts"]
    tournament_sizes = summary["tournament_sizes"]
    avg_fitness = summary["avg_fitness"]
    opt_data = summary["optimal_m_per_triangle"]
 
    lines = []
    lines.append("=" * 90)
    lines.append("TOURNAMENT M vs TRIANGLE COUNT — SUMMARY")
    lines.append("=" * 90)
    lines.append("")
 
    header = f"{'Triangles':>10}" + "".join(f"  M={m:<5}" for m in tournament_sizes) + "  │  Optimal M"
    lines.append(header)
    lines.append("-" * len(header))
 
    for t in triangle_counts:
        row = f"{t:>10}"
        for m in tournament_sizes:
            val = avg_fitness.get(str(t), {}).get(str(m), float("nan"))
            row += f"  {val:>6.4f} "
        opt = opt_data.get(str(t), {})
        row += f"  │  M={opt.get('optimal_m','?')}  (fit={opt.get('fitness', float('nan')):.4f})"
        lines.append(row)
 
    lines.append("")
    lines.append("Optimal M scaling:")
    for t in triangle_counts:
        opt = opt_data.get(str(t), {})
        lines.append(f"  {t:3d} triangles  →  optimal M = {opt.get('optimal_m','?')}")
    lines.append("=" * 90)
 
    text = "\n".join(lines)
    print(text)
 
    out = output_dir / "summary_table.txt"
    with open(out, "w") as f:
        f.write(text)
    print(f"\nSaved: {out}")
 
 
# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary", type=str, required=True,
        help="Path to summary.json produced by tournament_m_triangles_experiment.py",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing results_triangles_N.json files (for std dev bands). "
             "Defaults to same directory as summary.json.",
    )
    parser.add_argument(
        "--output", type=str, default="plots/tournament_m_triangles",
        help="Output directory for plots",
    )
    args = parser.parse_args()
 
    summary_path = Path(args.summary)
    results_dir = Path(args.results_dir) if args.results_dir else summary_path.parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"Loading summary: {summary_path}")
    summary = load_json(str(summary_path))
 
    triangle_counts = summary["triangle_counts"]
    print(f"Triangle counts : {triangle_counts}")
    print(f"Tournament sizes: {summary['tournament_sizes']}")
    print(f"Loading per-triangle JSONs from: {results_dir}")
 
    per_triangle = load_per_triangle_results(str(results_dir), triangle_counts)
    print()
 
    print("Generating plots...")
    plot_heatmap(summary, output_dir)
    plot_fitness_lines(summary, per_triangle, output_dir)
    plot_optimal_m_scaling(summary, output_dir)
    plot_individual_subplots(summary, per_triangle, output_dir)
    print_and_save_summary(summary, output_dir)
 
    print(f"\nAll plots saved to: {output_dir}")
 
 
if __name__ == "__main__":
    main()
 