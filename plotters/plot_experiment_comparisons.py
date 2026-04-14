"""
Boxplots from experiment_runner CSV output (matplotlib).

Visual style aligned with TPE1 scripts/plot_level_metrics.py (same palette, fonts, boxplot+jitter).

Features
--------
- All hyperparameter columns on the X axis (default) or a subset via --x-axis.
- Row filtering via --filter KEY=VALUE (repeatable) to fix parameters.
- Y-axis metrics: best_fitness, generations_run, elapsed_seconds (if present),
  and derived fitness_per_generation / fitness_per_second.
- --y-axis METRIC (repeatable) to restrict which Y metrics are plotted.
- --pairwise: one figure per pair of X-axis values (all C(n,2) pairs), with
  optional Mann-Whitney U p-value annotation if scipy is available.
- --add-stats: annotate mean ± std and CI below each box.
- --show-mean: overlay a diamond marker at the mean.
- --ci LEVEL: confidence level for the CI annotation (default 95).

Requires: pip install matplotlib numpy
Optional: pip install scipy  (for pairwise statistical tests)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import sys
from pathlib import Path
from typing import NamedTuple

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    from matplotlib.lines import Line2D
except ImportError as e:
    raise SystemExit(
        "matplotlib and numpy are required. Install with: pip install matplotlib numpy"
    ) from e

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Visual constants (aligned with TPE1 plot_level_metrics.py)
# ---------------------------------------------------------------------------
FIG_DPI = 144
FIG_SIZE = (11.0, 6.2)
SAVE_PAD_INCHES = 0.2

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

BOXPLOT_STYLE = {
    "box_face": "#4a90d9",
    "box_edge": "#2c6cb0",
    "median_color": "#c0392b",
    "whisker_color": "#343434",
    "cap_color": "#343434",
    "flier_color": "#95a5a6",
    "point_color": "#e67e22",
    "point_edge": "#c45f1a",
    "mean_color": "#27ae60",
}

# Palette for pairwise plots (two distinct colours)
PAIRWISE_COLORS = [
    {"box_face": "#4a90d9", "box_edge": "#2c6cb0"},
    {"box_face": "#e67e22", "box_edge": "#c45f1a"},
]

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
REQUIRED_METRIC_COLUMNS = ("best_fitness", "generations_run")
OPTIONAL_METRIC_COLUMNS = ("elapsed_seconds",)
DERIVED_METRIC_COLUMNS = ("fitness_per_generation", "fitness_per_second")
INTEGER_METRICS = frozenset({"generations_run"})

METRIC_YLABEL_ES: dict[str, str] = {
    "best_fitness": "Mejor fitness",
    "generations_run": "Generaciones ejecutadas",
    "elapsed_seconds": "Tiempo de ejecución (s)",
    "fitness_per_generation": "Fitness por generación",
    "fitness_per_second": "Fitness por segundo",
}

PARAM_LABEL_ES: dict[str, str] = {
    "selector": "Método de selección",
    "crossover": "Cruce",
    "fitness": "Función de aptitud",
    "survival_strategy": "Estrategia de supervivencia",
    "population_size": "Tamaño de población",
    "generations": "Generaciones (máx.)",
    "k": "k (padres/hijos por generación)",
    "mutation": "Mutación",
    "mutation_rate": "Tasa de mutación",
    "mutation_strength": "Fuerza de mutación",
    "triangles": "Cantidad de triángulos",
    "seed": "Semilla",
    "input_image": "Imagen de entrada",
    "output_image": "Imagen de salida",
    "output_dir": "Directorio de salida",
    "snapshot_interval": "Intervalo de capturas (generaciones)",
    "temperature": "Temperatura inicial (Boltzmann)",
    "temperature_min": "Temperatura mínima (Boltzmann)",
    "temperature_decay": "Decaimiento de temperatura (Boltzmann)",
    "target_fitness": "Fitness objetivo (parada temprana)",
    "convergence_window": "Ventana de convergencia (generaciones)",
    "convergence_delta": "Delta mínimo de mejora",
    "time_limit": "Límite de tiempo (s)",
}


def metric_ylabel_es(metric: str) -> str:
    return METRIC_YLABEL_ES.get(metric, metric.replace("_", " "))


def param_xlabel_es(param: str) -> str:
    return PARAM_LABEL_ES.get(param, param.replace("_", " "))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename_part(s: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    return s.strip("_") or "out"


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def parse_float_cell(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    return _try_float(raw)


def parse_int_cell(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def sort_category_labels(labels: list[str]) -> list[str]:
    if not labels:
        return labels
    if all(_try_float(s) is not None for s in labels):
        return sorted(labels, key=lambda s: float(s))
    return sorted(labels)


def unique_non_null(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v is None or v == "":
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# CSV loading, filtering, metric computation
# ---------------------------------------------------------------------------

def load_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("CSV has no header row.")
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    return rows, fieldnames


def validate_columns(fieldnames: list[str]) -> None:
    missing = [c for c in REQUIRED_METRIC_COLUMNS if c not in fieldnames]
    if missing:
        raise SystemExit(
            f"Missing required CSV columns: {missing}. Found: {fieldnames}"
        )


def apply_filters(
    rows: list[dict[str, str]],
    filters: dict[str, list[str]],
) -> list[dict[str, str]]:
    """Keep only rows where, for every filtered key, the row value is in the allowed set.

    Multiple values for the same key act as OR (the row matches if it equals any of them).
    Different keys act as AND (all conditions must hold).
    """
    if not filters:
        return rows
    return [
        r for r in rows
        if all(r.get(k, "") in allowed for k, allowed in filters.items())
    ]


def compute_derived_metrics(
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    """
    Compute derived metric columns and add them to each row.
    Returns updated rows and updated fieldnames list.
    """
    has_elapsed = "elapsed_seconds" in fieldnames
    new_fields = list(fieldnames)

    if "fitness_per_generation" not in new_fields:
        new_fields.append("fitness_per_generation")
    if has_elapsed and "fitness_per_second" not in new_fields:
        new_fields.append("fitness_per_second")

    updated: list[dict[str, str]] = []
    for row in rows:
        r = dict(row)

        bf = parse_float_cell(r.get("best_fitness"))
        gr = parse_float_cell(r.get("generations_run"))
        if bf is not None and gr is not None and gr > 0:
            r["fitness_per_generation"] = str(bf / gr)
        else:
            r["fitness_per_generation"] = ""

        if has_elapsed:
            el = parse_float_cell(r.get("elapsed_seconds"))
            if bf is not None and el is not None and el > 0:
                r["fitness_per_second"] = str(bf / el)
            else:
                r["fitness_per_second"] = ""

        updated.append(r)

    return updated, new_fields


def resolve_active_metrics(
    requested: list[str] | None,
    fieldnames: list[str],
) -> list[str]:
    """Return the list of metric columns to plot, in a fixed order."""
    all_metrics = list(REQUIRED_METRIC_COLUMNS)
    for m in OPTIONAL_METRIC_COLUMNS:
        if m in fieldnames:
            all_metrics.append(m)
    for m in DERIVED_METRIC_COLUMNS:
        if m in fieldnames:
            all_metrics.append(m)

    if not requested:
        return all_metrics

    unknown = [m for m in requested if m not in all_metrics]
    if unknown:
        raise SystemExit(
            f"Unknown --y-axis: {unknown}. Available: {all_metrics}"
        )
    return [m for m in all_metrics if m in requested]


def param_columns(fieldnames: list[str]) -> list[str]:
    all_metric_cols = set(REQUIRED_METRIC_COLUMNS) | set(OPTIONAL_METRIC_COLUMNS) | set(DERIVED_METRIC_COLUMNS)
    return [c for c in fieldnames if c not in all_metric_cols]


def resolve_x_axis_columns(
    requested: list[str] | None,
    fieldnames: list[str],
) -> list[str]:
    all_params = param_columns(fieldnames)
    if not requested:
        return all_params
    unknown = [c for c in requested if c not in all_params]
    if unknown:
        raise SystemExit(
            f"Unknown --x-axis: {unknown}. Plottable columns: {all_params}"
        )
    bad_metrics = [c for c in requested if c in set(REQUIRED_METRIC_COLUMNS) | set(OPTIONAL_METRIC_COLUMNS) | set(DERIVED_METRIC_COLUMNS)]
    if bad_metrics:
        raise SystemExit(f"--x-axis cannot include metric columns: {bad_metrics}")
    return list(requested)


def collect_groups(
    rows: list[dict[str, str]],
    param: str,
    metric: str,
) -> tuple[list[list[float]], list[str]]:
    """One list of sample values per distinct param value (ordered)."""
    by_val: dict[str, list[float]] = {}
    for r in rows:
        key = r.get(param)
        if key is None or key == "":
            continue
        raw = r.get(metric)
        if metric in INTEGER_METRICS:
            v = parse_int_cell(raw)
            fv = float(v) if v is not None else None
        else:
            fv = parse_float_cell(raw)
        if fv is None:
            continue
        by_val.setdefault(key, []).append(fv)

    labels = sort_category_labels(list(by_val.keys()))
    data = [by_val[lab] for lab in labels]
    return data, labels


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

class GroupStats(NamedTuple):
    mean: float
    std: float
    ci_lo: float
    ci_hi: float
    n: int


def compute_stats(values: list[float], ci_level: float = 0.95) -> GroupStats | None:
    if not values:
        return None
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if n > 1 and _SCIPY_AVAILABLE:
        lo, hi = _scipy_stats.t.interval(ci_level, df=n - 1, loc=mean, scale=std / math.sqrt(n))
    elif n > 1:
        # Normal approximation
        from scipy.stats import norm  # noqa: F401 (never reached if scipy missing)
        z = 1.96 if abs(ci_level - 0.95) < 1e-6 else 2.576
        margin = z * std / math.sqrt(n)
        lo, hi = mean - margin, mean + margin
    else:
        lo, hi = mean, mean
    return GroupStats(mean=mean, std=std, ci_lo=float(lo), ci_hi=float(hi), n=n)


def mannwhitney_pvalue(a: list[float], b: list[float]) -> float | None:
    if not _SCIPY_AVAILABLE or len(a) < 2 or len(b) < 2:
        return None
    try:
        result = _scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(result.pvalue)
    except Exception:
        return None


def pvalue_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_axes_style(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(STYLE["figure_bg"])
    ax.set_facecolor(STYLE["axes_bg"])
    ax.grid(axis="y", which="major", linestyle="-", linewidth=0.6,
            alpha=0.55, color=STYLE["grid"], zorder=0)
    ax.minorticks_on()
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.4,
            alpha=0.45, color=STYLE["grid_minor"], zorder=0)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="both", colors=STYLE["text_axis"])
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(STYLE["text_axis"])


def _apply_boxplot_style(
    bp: dict,
    face_color: str = BOXPLOT_STYLE["box_face"],
    edge_color: str = BOXPLOT_STYLE["box_edge"],
) -> None:
    for box in bp["boxes"]:
        box.set_facecolor(face_color)
        box.set_edgecolor(edge_color)
        box.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color(BOXPLOT_STYLE["median_color"])
        median.set_linewidth(2.0)
    for whisker in bp["whiskers"]:
        whisker.set_color(BOXPLOT_STYLE["whisker_color"])
        whisker.set_linewidth(0.9)
    for cap in bp["caps"]:
        cap.set_color(BOXPLOT_STYLE["cap_color"])
        cap.set_linewidth(0.9)


def _add_jitter(
    ax: plt.Axes,
    positions: list[int],
    data: list[list[float]],
    rng: np.random.Generator,
    color: str = BOXPLOT_STYLE["point_color"],
    edge_color: str = BOXPLOT_STYLE["point_edge"],
) -> None:
    for i, vals in enumerate(data):
        if not vals:
            continue
        ax.scatter(
            np.full(len(vals), positions[i]),
            vals,
            s=20,
            color=color,
            edgecolors=edge_color,
            linewidths=0.5,
            alpha=0.85,
            zorder=3,
        )


def _add_mean_markers(
    ax: plt.Axes,
    positions: list[int],
    data: list[list[float]],
) -> None:
    for i, vals in enumerate(data):
        if not vals:
            continue
        mean_val = float(np.mean(vals))
        ax.plot(
            positions[i],
            mean_val,
            marker="D",
            markersize=7,
            color=BOXPLOT_STYLE["mean_color"],
            markeredgecolor="#1a7a44",
            markeredgewidth=0.8,
            zorder=5,
            label="_nolegend_",
        )


def _add_stats_annotations(
    ax: plt.Axes,
    positions: list[int],
    data: list[list[float]],
    ci_level: float,
    y_min: float,
    y_range: float,
) -> None:
    """Place stats inside the plot, above the upper whisker cap of each box."""
    for i, vals in enumerate(data):
        if not vals:
            continue
        s = compute_stats(vals, ci_level)
        if s is None:
            continue
        ci_pct = int(round(ci_level * 100))
        # Upper whisker cap: min(max(vals), Q3 + 1.5*IQR)
        q3 = float(np.percentile(vals, 75))
        iqr = q3 - float(np.percentile(vals, 25))
        cap = min(max(vals), q3 + 1.5 * iqr)
        text_y = cap + y_range * 0.03
        text = (
            f"n={s.n}  μ={s.mean:.4g}\n"
            f"σ={s.std:.4g}  IC{ci_pct}%:[{s.ci_lo:.4g},{s.ci_hi:.4g}]"
        )
        ax.text(
            positions[i],
            text_y,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
            color=STYLE["stats_text"],
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.75,
                edgecolor=STYLE["grid"],
                linewidth=0.5,
            ),
            zorder=6,
        )


def _set_y_formatter(ax: plt.Axes, metric: str) -> None:
    if metric in INTEGER_METRICS:
        ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.6g}"))


def plot_all_methods(
    data_per_config: list[list[float]],
    labels: list[str],
    title: str,
    ylabel: str,
    metric: str,
    xlabel: str | None = None,
    show_mean: bool = False,
    add_stats: bool = False,
    ci_level: float = 0.95,
) -> plt.Figure:
    """Boxplot showing all groups side by side."""
    n = len(labels)
    fig_w = max(11.0, 0.9 * n + 3.0)
    fig_h = FIG_SIZE[1]

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIG_DPI)
    _setup_axes_style(fig, ax)

    if n == 0 or all(len(d) == 0 for d in data_per_config):
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                transform=ax.transAxes, color=STYLE["text_axis"])
        ax.set_title(title, fontsize=13, fontweight="600",
                     color=STYLE["text_title"], pad=14)
        if xlabel:
            ax.set_xlabel(xlabel, color=STYLE["text_axis"])
        fig.subplots_adjust(left=0.09, right=0.9, top=0.88, bottom=0.14)
        return fig

    positions = list(range(1, n + 1))
    bp = ax.boxplot(data_per_config, positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False, zorder=2)
    _apply_boxplot_style(bp)

    rng = np.random.default_rng(42)
    _add_jitter(ax, positions, data_per_config, rng)

    if show_mean:
        _add_mean_markers(ax, positions, data_per_config)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    if xlabel:
        ax.set_xlabel(xlabel, color=STYLE["text_axis"])
    ax.set_ylabel(ylabel, color=STYLE["text_axis"])
    _set_y_formatter(ax, metric)

    all_vals = [v for d in data_per_config for v in d]
    y_min = min(all_vals) if all_vals else 0.0
    y_max = max(all_vals) if all_vals else 1.0
    y_range = y_max - y_min or 1.0

    if add_stats:
        _add_stats_annotations(ax, positions, data_per_config, ci_level, y_min, y_range)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=BOXPLOT_STYLE["median_color"], linewidth=1.8, label="Mediana"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BOXPLOT_STYLE["point_color"],
               markersize=6, label="Observaciones"),
    ]
    if show_mean:
        legend_handles.append(
            Line2D([0], [0], marker="D", color="w",
                   markerfacecolor=BOXPLOT_STYLE["mean_color"],
                   markersize=7, label="Media")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.8, edgecolor=STYLE["grid"])

    ax.set_title(title, fontsize=13, fontweight="600",
                 color=STYLE["text_title"], pad=14)

    fig.subplots_adjust(left=0.09, right=0.93, top=0.88, bottom=0.22)
    return fig


def plot_pairwise_comparison(
    data_a: list[float],
    data_b: list[float],
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    metric: str,
    xlabel: str | None = None,
    show_mean: bool = False,
    add_stats: bool = False,
    ci_level: float = 0.95,
) -> plt.Figure:
    """Boxplot comparing exactly two groups with optional statistical annotation."""
    fig_h = FIG_SIZE[1]

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(5.5, fig_h), dpi=FIG_DPI)
    _setup_axes_style(fig, ax)

    positions = [1, 1.6]
    data = [data_a, data_b]
    colors = PAIRWISE_COLORS

    for idx, (vals, pos, clr) in enumerate(zip(data, positions, colors)):
        if not vals:
            continue
        bp = ax.boxplot([vals], positions=[pos], widths=0.5, patch_artist=True,
                        showfliers=False, zorder=2)
        _apply_boxplot_style(bp, face_color=clr["box_face"], edge_color=clr["box_edge"])

    rng = np.random.default_rng(42)
    point_colors = [
        (PAIRWISE_COLORS[0]["box_face"], PAIRWISE_COLORS[0]["box_edge"]),
        (PAIRWISE_COLORS[1]["box_face"], PAIRWISE_COLORS[1]["box_edge"]),
    ]
    for i, (vals, pos) in enumerate(zip(data, positions)):
        if not vals:
            continue
        ax.scatter(
            np.full(len(vals), pos),
            vals,
            s=20,
            color=point_colors[i][0],
            edgecolors=point_colors[i][1],
            linewidths=0.5,
            alpha=0.80,
            zorder=3,
        )

    if show_mean:
        _add_mean_markers(ax, positions, data)

    ax.set_xticks(positions)
    ax.set_xticklabels([label_a, label_b], rotation=20, ha="right")
    ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=STYLE["text_axis"])
    ax.set_ylabel(ylabel, color=STYLE["text_axis"])
    _set_y_formatter(ax, metric)

    all_vals = [v for d in data for v in d]
    y_min = min(all_vals) if all_vals else 0.0
    y_max = max(all_vals) if all_vals else 1.0
    y_range_val = y_max - y_min or 1.0

    if add_stats:
        _add_stats_annotations(ax, positions, data, ci_level, y_min, y_range_val)

    legend_handles = [
        Line2D([0], [0], color=BOXPLOT_STYLE["median_color"], linewidth=1.8, label="Mediana"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PAIRWISE_COLORS[0]["box_face"],
               markersize=7, label=label_a),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PAIRWISE_COLORS[1]["box_face"],
               markersize=7, label=label_b),
    ]
    if show_mean:
        legend_handles.append(
            Line2D([0], [0], marker="D", color="w",
                   markerfacecolor=BOXPLOT_STYLE["mean_color"],
                   markersize=7, label="Media")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.8, edgecolor=STYLE["grid"])

    ax.set_title(title, fontsize=12, fontweight="600",
                 color=STYLE["text_title"], pad=14)

    fig.subplots_adjust(left=0.11, right=0.93, top=0.88, bottom=0.22)
    return fig


# ---------------------------------------------------------------------------
# Scatter plot: generations_run vs efficiency metric, coloured by param
# ---------------------------------------------------------------------------

# Colour palette for up to 10 groups (selector-level colours)
_SCATTER_PALETTE = [
    "#4a90d9", "#e67e22", "#27ae60", "#c0392b", "#8e44ad",
    "#16a085", "#d35400", "#2980b9", "#f39c12", "#7f8c8d",
]


def _normalise_data(data: list[list[float]]) -> list[list[float]]:
    """Min-max normalise each value across all groups to [0, 1]."""
    all_vals = [v for d in data for v in d]
    if not all_vals:
        return data
    mn, mx = min(all_vals), max(all_vals)
    rng = mx - mn or 1.0
    return [[(v - mn) / rng for v in d] for d in data]


def plot_dual_boxplot(
    data_left: list[list[float]],
    data_right: list[list[float]],
    labels: list[str],
    metric_left: str,
    metric_right: str,
    title: str,
) -> plt.Figure:
    """Grouped boxplot: for each category, two boxes side by side (one per metric).
    Both metrics are normalised to [0, 1] so they share the same Y axis.
    """
    n = len(labels)
    fig_w = max(11.0, 1.2 * n + 3.0)

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(fig_w, FIG_SIZE[1]), dpi=FIG_DPI)
    _setup_axes_style(fig, ax)

    norm_l = _normalise_data(data_left)
    norm_r = _normalise_data(data_right)

    gap = 0.28      # half-distance between the two boxes of the same group
    width = 0.24

    color_l = "#4a90d9"
    edge_l  = "#2c6cb0"
    color_r = "#e67e22"
    edge_r  = "#c45f1a"

    group_centers = list(range(1, n + 1))
    pos_l = [c - gap for c in group_centers]
    pos_r = [c + gap for c in group_centers]

    rng = np.random.default_rng(42)

    for pos_list, data, fc, ec in (
        (pos_l, norm_l, color_l, edge_l),
        (pos_r, norm_r, color_r, edge_r),
    ):
        valid = [(p, d) for p, d in zip(pos_list, data) if d]
        if not valid:
            continue
        bp = ax.boxplot(
            [d for _, d in valid],
            positions=[p for p, _ in valid],
            widths=width,
            patch_artist=True,
            showfliers=False,
            zorder=2,
        )
        _apply_boxplot_style(bp, face_color=fc, edge_color=ec)
        _add_jitter(ax, [p for p, _ in valid], [d for _, d in valid], rng,
                    color=fc, edge_color=ec)

    # X-axis: group labels at center positions
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlim(0.4, n + 0.6)
    ax.set_ylabel("Valor normalizado [0 → 1]", color=STYLE["text_axis"])
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

    # Thin vertical separators between groups
    for c in group_centers[:-1]:
        ax.axvline(c + 0.5, color=STYLE["grid"], linewidth=0.5,
                   linestyle=":", alpha=0.6, zorder=0)

    legend_handles = [
        Line2D([0], [0], color=BOXPLOT_STYLE["median_color"],
               linewidth=1.8, label="Mediana"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.85, edgecolor=STYLE["grid"])

    ax.set_title(title, fontsize=13, fontweight="600",
                 color=STYLE["text_title"], pad=14)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.22)
    return fig


def plot_scatter_efficiency(
    rows: list[dict[str, str]],
    param: str,
    x_metric: str,
    y_metric: str,
    title: str,
    stratum_value: str | None = None,
) -> plt.Figure:
    """Cleveland dot plot: one row per group, two dots per row (x and y metric,
    both normalised to [0,1]). Rows sorted by y_metric descending.
    Raw mean values printed next to each dot.
    """
    groups: dict[str, tuple[list[float], list[float]]] = {}
    for r in rows:
        key = r.get(param)
        if not key:
            continue
        xv = parse_float_cell(r.get(x_metric))
        yv = parse_float_cell(r.get(y_metric))
        if xv is None or yv is None:
            continue
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].append(xv)
        groups[key][1].append(yv)

    labels = sort_category_labels(list(groups.keys()))
    means_x = {lb: float(np.mean(groups[lb][0])) for lb in labels if groups[lb][0]}
    means_y = {lb: float(np.mean(groups[lb][1])) for lb in labels if groups[lb][1]}
    valid = [lb for lb in labels if lb in means_x and lb in means_y]
    if not valid:
        with plt.rc_context(PLOT_RC):
            fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
        return fig

    # Sort rows by y_metric descending (best efficiency at top)
    valid.sort(key=lambda lb: means_y[lb], reverse=True)

    # Normalise both metrics to [0, 1]
    all_x = [means_x[lb] for lb in valid]
    all_y = [means_y[lb] for lb in valid]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    rng_x = max_x - min_x or 1.0
    rng_y = max_y - min_y or 1.0
    norm = lambda v, mn, rng: (v - mn) / rng  # noqa: E731

    n = len(valid)
    fig_h = max(4.0, 0.55 * n + 1.8)
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(9.0, fig_h), dpi=FIG_DPI)
    _setup_axes_style(fig, ax)

    y_positions = list(range(n))
    x_color = "#4a90d9"
    y_color = "#e67e22"

    for i, lb in enumerate(valid):
        ypos = n - 1 - i  # top = best

        nx = norm(means_x[lb], min_x, rng_x)
        ny = norm(means_y[lb], min_y, rng_y)

        # Connecting segment between the two dots
        ax.plot([nx, ny], [ypos, ypos], color=STYLE["grid"],
                linewidth=1.0, zorder=1)

        # x_metric dot
        ax.scatter(nx, ypos, s=80, color=x_color, edgecolors="white",
                   linewidths=0.8, zorder=3)
        ax.text(nx, ypos + 0.28, f"{means_x[lb]:.4g}",
                ha="center", va="bottom", fontsize=7, color=x_color)

        # y_metric dot
        ax.scatter(ny, ypos, s=80, marker="D", color=y_color,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(ny, ypos + 0.28, f"{means_y[lb]:.4g}",
                ha="center", va="bottom", fontsize=7, color=y_color)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(list(reversed(valid)), fontsize=9)
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["mín", "25%", "50%", "75%", "máx"], fontsize=8,
                       color=STYLE["text_axis"])
    ax.set_xlabel("Valor relativo (normalizado al rango del grupo)", fontsize=9,
                  color=STYLE["text_axis"])
    ax.grid(axis="x", which="major", linestyle="--", linewidth=0.5,
            alpha=0.4, color=STYLE["grid"], zorder=0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    from matplotlib.lines import Line2D  # noqa: PLC0415
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=x_color,
               markersize=8, label=metric_ylabel_es(x_metric)),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=y_color,
               markersize=8, label=metric_ylabel_es(y_metric)),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.85, edgecolor=STYLE["grid"])

    suffix = f" (función de aptitud: {stratum_value})" if stratum_value else ""
    ax.set_title(f"{title}{suffix}", fontsize=13, fontweight="600",
                 color=STYLE["text_title"], pad=14)

    fig.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.12)
    return fig


# ---------------------------------------------------------------------------
# Convergence curve plot
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    history_csv: Path,
    out_dir: Path,
    filters: dict[str, list[str]] | None = None,
    color_by: str = "selector",
    x_axis: str = "generation",
    show_seeds: bool = False,
) -> list[Path]:
    """Line plot of best_fitness vs generation or elapsed_seconds.
    Mean across seeds: thick opaque line.
    Individual seeds: thin semi-transparent lines (only if show_seeds=True).
    x_axis: 'generation' or 'elapsed_seconds'.
    """
    from collections import defaultdict

    rows, fieldnames = load_rows(history_csv)
    if "best_fitness" not in fieldnames:
        print("History CSV must have a 'best_fitness' column.")
        return []
    if x_axis not in fieldnames:
        print(f"Column '{x_axis}' not found in history CSV. "
              f"Available: {fieldnames}")
        return []

    if filters:
        rows = apply_filters(rows, filters)
    if not rows:
        print("No rows after filtering.")
        return []

    if color_by not in fieldnames:
        print(f"Column '{color_by}' not found in history CSV.")
        return []

    seed_col = "seed" if "seed" in fieldnames else None

    # groups[label][seed] = list of (x_val, fitness)
    groups: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        key = r.get(color_by, "")
        if not key:
            continue
        seed = r.get(seed_col, "0") if seed_col else "0"
        xv = parse_float_cell(r.get(x_axis))
        fit = parse_float_cell(r.get("best_fitness"))
        if xv is None or fit is None:
            continue
        groups[key][seed].append((xv, fit))

    labels = sort_category_labels(list(groups.keys()))
    if not labels:
        return []

    x_label = "Generación" if x_axis == "generation" else "Tiempo (s)"

    plt.ioff()
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    _setup_axes_style(fig, ax)

    for i, label in enumerate(labels):
        color = _SCATTER_PALETTE[i % len(_SCATTER_PALETTE)]
        seed_data = groups[label]
        all_series: list[list[tuple[float, float]]] = [
            sorted(seed_data[s], key=lambda p: p[0]) for s in sorted(seed_data)
        ]

        if x_axis == "elapsed_seconds":
            # Interpolate each seed to a common time grid (step function: fitness only increases)
            x_min = min(pts[0][0] for pts in all_series if pts)
            x_max = max(pts[-1][0] for pts in all_series if pts)
            grid = np.linspace(x_min, x_max, 400)

            interp_matrix = []
            for pts in all_series:
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                interp_y = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
                interp_matrix.append(interp_y)
                if show_seeds:
                    ax.plot(grid, interp_y, color=color, linewidth=0.8, alpha=0.25, zorder=2)

            if len(interp_matrix) > 1:
                mean_fits = np.array(interp_matrix).mean(axis=0)
                ax.plot(grid, mean_fits, color=color, linewidth=2.2,
                        alpha=0.95, zorder=4, label=label)
            else:
                ax.plot(grid, interp_matrix[0], color=color, linewidth=2.0,
                        alpha=0.9, zorder=4, label=label)
        else:
            # Generation mode: seeds share the same x values, no interpolation needed
            if show_seeds:
                for pts in all_series:
                    ax.plot(
                        [p[0] for p in pts], [p[1] for p in pts],
                        color=color, linewidth=0.8, alpha=0.25, zorder=2,
                    )

            if len(all_series) > 1:
                x_fitness: dict[float, list[float]] = defaultdict(list)
                for series in all_series:
                    for xv, fit in series:
                        x_fitness[xv].append(fit)
                mean_xs = sorted(x_fitness)
                mean_fits_gen = [float(np.mean(x_fitness[xv])) for xv in mean_xs]
                ax.plot(mean_xs, mean_fits_gen, color=color, linewidth=2.2,
                        alpha=0.95, zorder=4, label=label)
            elif all_series:
                pts = all_series[0]
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=color, linewidth=2.0, alpha=0.9, zorder=4, label=label)

    ax.set_xlabel(x_label, color=STYLE["text_axis"])
    ax.set_ylabel("Mejor fitness", color=STYLE["text_axis"])
    ax.legend(
        title=param_xlabel_es(color_by), fontsize=8, title_fontsize=8,
        framealpha=0.85, edgecolor=STYLE["grid"], loc="lower right",
    )

    title = f"Curvas de convergencia por {param_xlabel_es(color_by).lower()}"
    ax.set_title(title, fontsize=13, fontweight="600",
                 color=STYLE["text_title"], pad=14)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.12)

    out_dir.mkdir(parents=True, exist_ok=True)
    x_suffix = "_vs_time" if x_axis == "elapsed_seconds" else ""
    dest = out_dir / f"convergence{x_suffix}_by_{_safe_filename_part(color_by)}.png"
    save_figure(fig, dest)
    plt.close(fig)
    print(f"Saved: {dest}")
    return [dest]


def build_strata(
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> list[tuple[str | None, list[dict[str, str]]]]:
    """
    If `fitness` varies, one stratum per distinct fitness value.
    Otherwise a single stratum (suffix None -> omitted in filenames).
    """
    if not rows or "fitness" not in fieldnames:
        return [(None, rows)]
    values = unique_non_null([r.get("fitness") for r in rows])
    if len(values) <= 1:
        return [(None, rows)]
    ordered = sort_category_labels(values)
    return [(v, [r for r in rows if (r.get("fitness") or "") == v]) for v in ordered]


# ---------------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------------

def output_basename(metric: str, param: str, stratum_value: str | None,
                    label_a: str | None = None, label_b: str | None = None) -> str:
    core = f"boxplot_{metric}_by_{param}"
    if label_a is not None and label_b is not None:
        core += f"__pair_{_safe_filename_part(label_a)}_vs_{_safe_filename_part(label_b)}"
    if stratum_value is not None:
        core += f"__fitness_{_safe_filename_part(stratum_value)}"
    return f"{_safe_filename_part(core)}.png"


def build_title(
    param: str,
    metric: str,
    stratum_value: str | None,
    multi_fitness_grid: bool,
    label_a: str | None = None,
    label_b: str | None = None,
    n_labels: int = 0,
) -> str:
    ylabel = metric_ylabel_es(metric)
    if n_labels == 1 and label_a is not None:
        # Single group: "<method> — <metric>"
        base = f"{label_a} — {ylabel}"
    elif label_a is not None and label_b is not None:
        # Two groups (pairwise or all-methods with 2): "<A> vs. <B> — <metric>"
        base = f"{label_a} vs. {label_b} — {ylabel}"
    else:
        # Three or more: "Comparación por <param> — <metric>"
        base = f"Comparación por {param_xlabel_es(param).lower()} — {ylabel}"
    if multi_fitness_grid and stratum_value is not None:
        base += f" (función de aptitud: {stratum_value})"
    return base


# ---------------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
        facecolor=fig.get_facecolor(),
        format=path.suffix.lstrip(".").lower() or "png",
    )


# ---------------------------------------------------------------------------
# Main run logic
# ---------------------------------------------------------------------------

def run(
    csv_path: Path,
    out_dir: Path,
    filters: dict[str, list[str]] | None = None,
    x_axis_columns: list[str] | None = None,
    y_axis_metrics: list[str] | None = None,
    pairwise: bool = False,
    show_mean: bool = False,
    add_stats: bool = False,
    ci_level: float = 0.95,
    scatter: bool = False,
) -> list[Path]:
    rows, fieldnames = load_rows(csv_path)
    validate_columns(fieldnames)

    # Apply filters
    if filters:
        rows = apply_filters(rows, filters)
        if not rows:
            print(
                f"Warning: No rows remain after applying filters {filters}. "
                "Check filter values match the CSV content."
            )
            return []

    # Compute derived metrics
    rows, fieldnames = compute_derived_metrics(rows, fieldnames)

    params = resolve_x_axis_columns(x_axis_columns, fieldnames)
    metrics = resolve_active_metrics(y_axis_metrics, fieldnames)

    if not rows:
        print("Warning: CSV has no data rows.")
        return []

    strata = build_strata(rows, fieldnames)
    multi_fitness = len(strata) > 1 or (
        "fitness" in fieldnames
        and len(unique_non_null([r.get("fitness") for r in rows])) > 1
    )

    written: list[Path] = []
    plt.ioff()

    for stratum_value, stratum_rows in strata:
        suffix_stratum = stratum_value if multi_fitness else None

        for param in params:
            labels_all = sort_category_labels(
                unique_non_null([r.get(param) for r in stratum_rows])
            )
            if len(labels_all) < 1:
                continue

            for metric in metrics:
                data, labels = collect_groups(stratum_rows, param, metric)
                if len(labels) < 1 or all(len(d) == 0 for d in data):
                    continue

                n_groups = len(labels)
                ylabel = metric_ylabel_es(metric)

                # xlabel and title depend on number of groups
                if n_groups == 1:
                    xlabel = None
                    title = build_title(
                        param, metric, stratum_value, multi_fitness,
                        label_a=labels[0], n_labels=1,
                    )
                elif n_groups == 2:
                    xlabel = None
                    title = build_title(
                        param, metric, stratum_value, multi_fitness,
                        label_a=labels[0], label_b=labels[1],
                    )
                else:
                    xlabel = param_xlabel_es(param)
                    title = build_title(param, metric, stratum_value, multi_fitness)

                # --- All-methods plot ---
                fig = plot_all_methods(
                    data, labels, title, ylabel, metric,
                    xlabel=xlabel,
                    show_mean=show_mean,
                    add_stats=add_stats,
                    ci_level=ci_level,
                )
                name = output_basename(metric, param, suffix_stratum)
                dest = out_dir / name
                save_figure(fig, dest)
                plt.close(fig)
                written.append(dest)
                print(f"Saved: {dest}")

                # --- Pairwise plots ---
                if pairwise:
                    # All subsets of size 2, preserving sorted order
                    for la, lb in itertools.combinations(labels, 2):
                        idx_a = labels.index(la)
                        idx_b = labels.index(lb)
                        da, db = data[idx_a], data[idx_b]
                        if not da or not db:
                            continue

                        pw_title = build_title(
                            param, metric, stratum_value, multi_fitness,
                            label_a=la, label_b=lb,
                        )
                        pw_fig = plot_pairwise_comparison(
                            da, db, la, lb,
                            title=pw_title,
                            ylabel=ylabel,
                            metric=metric,
                            xlabel=xlabel,
                            show_mean=show_mean,
                            add_stats=add_stats,
                            ci_level=ci_level,
                        )
                        pw_name = output_basename(metric, param, suffix_stratum,
                                                  label_a=la, label_b=lb)
                        pw_dest = out_dir / pw_name
                        save_figure(pw_fig, pw_dest)
                        plt.close(pw_fig)
                        written.append(pw_dest)
                        print(f"Saved: {pw_dest}")

    # --- Dual boxplots: two related metrics side by side ---
    if scatter:
        efficiency_pairs = [
            ("generations_run", "fitness_per_generation"),
        ]
        if "elapsed_seconds" in fieldnames:
            efficiency_pairs += [
                ("elapsed_seconds", "fitness_per_second"),
                ("generations_run", "elapsed_seconds"),
            ]

        for param in params:
            labels_all = sort_category_labels(
                unique_non_null([r.get(param) for r in rows])
            )
            if len(labels_all) < 2:
                continue

            for stratum_value, stratum_rows in strata:
                suffix_stratum = stratum_value if multi_fitness else None

                for met_l, met_r in efficiency_pairs:
                    if met_l not in fieldnames or met_r not in fieldnames:
                        continue
                    data_l, labels_l = collect_groups(stratum_rows, param, met_l)
                    data_r, labels_r = collect_groups(stratum_rows, param, met_r)
                    # Align labels (use intersection in sorted order)
                    shared = [lb for lb in labels_l if lb in labels_r]
                    if not shared:
                        continue
                    dl = [data_l[labels_l.index(lb)] for lb in shared]
                    dr = [data_r[labels_r.index(lb)] for lb in shared]
                    suffix_txt = (f" (función de aptitud: {suffix_stratum})"
                                  if suffix_stratum else "")
                    dual_title = (f"{metric_ylabel_es(met_l)} vs "
                                  f"{metric_ylabel_es(met_r)}{suffix_txt}")
                    fig = plot_dual_boxplot(
                        dl, dr, shared, met_l, met_r, dual_title,
                    )
                    name_core = f"dual_{met_l}_vs_{met_r}_by_{param}"
                    if suffix_stratum:
                        name_core += f"__fitness_{_safe_filename_part(suffix_stratum)}"
                    dest = out_dir / f"{_safe_filename_part(name_core)}.png"
                    save_figure(fig, dest)
                    plt.close(fig)
                    written.append(dest)
                    print(f"Saved: {dest}")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_filter(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--filter must be in KEY=VALUE format, got: {raw!r}"
        )
    k, _, v = raw.partition("=")
    return k.strip(), v.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Boxplots from experiment_runner.py CSV.\n"
            "Supports filtering, pairwise comparisons, derived metrics and statistical annotations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        required=False,
        default=None,
        type=Path,
        help="Path to CSV produced by experiment_runner.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for PNG figures (not required with --list-columns)",
    )
    parser.add_argument(
        "--x-axis",
        action="append",
        dest="x_axes",
        metavar="COL",
        help=(
            "CSV column for categories on the X axis (repeat for several). "
            "Default: every hyperparameter column. Example: --x-axis selector"
        ),
    )
    parser.add_argument(
        "--y-axis",
        action="append",
        dest="y_axes",
        metavar="METRIC",
        help=(
            "Metric to plot on the Y axis (repeat for several). "
            "Default: all available. "
            "Choices: best_fitness, generations_run, elapsed_seconds, "
            "fitness_per_generation, fitness_per_second"
        ),
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Restrict plots to these selection methods only (space-separated). "
            "Default: all selectors present in the CSV. "
            "Example: --selectors elite roulette boltzmann"
        ),
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="KEY=VALUE",
        type=_parse_filter,
        help=(
            "Filter rows to a specific parameter value before plotting. "
            "Repeat for multiple filters. "
            "Example: --filter mutation=multigen_limited --filter crossover=annular"
        ),
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help=(
            "Also generate one figure per pair of X-axis values (all C(n,2) pairs). "
            "Adds Mann-Whitney U p-value annotation when scipy is available."
        ),
    )
    parser.add_argument(
        "--show-mean",
        action="store_true",
        help="Overlay a diamond marker at the mean on each box.",
    )
    parser.add_argument(
        "--add-stats",
        action="store_true",
        help="Annotate each box with n, mean ± std, and confidence interval.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        metavar="LEVEL",
        help="Confidence level for CI annotation (default: 0.95). Example: --ci 0.99",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help=(
            "Generate scatter plots linking generations_run with fitness_per_generation "
            "(and elapsed_seconds with fitness_per_second if available). "
            "Each point is one run, coloured by the X-axis parameter."
        ),
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to the _history.csv produced by experiment_runner. "
            "When provided, generates convergence curve plots instead of boxplots."
        ),
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="selector",
        metavar="COL",
        help="Column used to colour convergence curves (default: selector).",
    )
    parser.add_argument(
        "--convergence-x",
        type=str,
        default="generation",
        choices=["generation", "elapsed_seconds"],
        metavar="AXIS",
        help=(
            "X axis for convergence curves: 'generation' (default) or "
            "'elapsed_seconds' (requires elapsed_seconds in history CSV)."
        ),
    )
    parser.add_argument(
        "--show-seeds",
        action="store_true",
        help="Show individual seed lines in convergence plots (default: only mean).",
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print plottable column names from the CSV header and exit.",
    )
    args = parser.parse_args()

    if args.csv is None and args.history_csv is None and not args.list_columns:
        parser.error("--csv is required unless using --history-csv or --list-columns")

    csv_path = args.csv.resolve() if args.csv else None
    if csv_path is not None and not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.list_columns:
        if csv_path is None:
            parser.error("--csv is required with --list-columns")
        rows, fieldnames = load_rows(csv_path)
        validate_columns(fieldnames)
        rows, fieldnames = compute_derived_metrics(rows, fieldnames)
        print("Hyperparameter columns (use with --x-axis):")
        for c in param_columns(fieldnames):
            print(f"  {c}")
        print("\nMetric columns (use with --y-axis):")
        for c in fieldnames:
            if c in set(REQUIRED_METRIC_COLUMNS) | set(OPTIONAL_METRIC_COLUMNS) | set(DERIVED_METRIC_COLUMNS):
                print(f"  {c}")
        return

    if args.out is None:
        parser.error("--out is required unless using --list-columns")

    if not _SCIPY_AVAILABLE:
        print(
            "Note: scipy not found — statistical tests (Mann-Whitney U) and exact "
            "CIs will be skipped. Install with: pip install scipy"
        )

    out_dir = args.out.resolve()
    filters: dict[str, list[str]] = {}
    for k, v in (args.filters or []):
        filters.setdefault(k, []).append(v)
    if args.selectors:
        filters.setdefault("selector", []).extend(args.selectors)

    # Convergence mode
    if args.history_csv is not None:
        hist_path = args.history_csv.resolve()
        if not hist_path.is_file():
            print(f"History CSV not found: {hist_path}", file=sys.stderr)
            sys.exit(1)
        paths = plot_convergence_curves(
            history_csv=hist_path,
            out_dir=out_dir,
            filters=filters,
            color_by=args.color_by,
            x_axis=args.convergence_x,
            show_seeds=args.show_seeds,
        )
        if paths:
            print(f"\nTotal: {len(paths)} figure(s) written to {out_dir}")
        return

    if csv_path is None:
        parser.error("--csv is required for boxplot mode")

    paths = run(
        csv_path=csv_path,
        out_dir=out_dir,
        filters=filters,
        x_axis_columns=args.x_axes,
        y_axis_metrics=args.y_axes,
        pairwise=args.pairwise,
        show_mean=args.show_mean,
        add_stats=args.add_stats,
        ci_level=args.ci,
        scatter=args.scatter,
    )

    if paths:
        print(f"\nTotal: {len(paths)} figure(s) written to {out_dir}")
    else:
        print("No figures written (check columns and that parameters vary across rows).")


if __name__ == "__main__":
    main()
