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
        fig, ax = plt.subplots(figsize=(8.0, fig_h), dpi=FIG_DPI)
    _setup_axes_style(fig, ax)

    positions = [1, 2]
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

    # Statistical test annotation
    pvalue = mannwhitney_pvalue(data_a, data_b)
    if pvalue is not None:
        stars = pvalue_stars(pvalue)
        y_top = max(
            (max(d) if d else 0.0) for d in data
        )
        all_vals = [v for d in data for v in d]
        y_range = (max(all_vals) - min(all_vals)) if all_vals else 1.0
        bar_y = y_top + y_range * 0.08
        ax.annotate(
            "",
            xy=(2, bar_y),
            xytext=(1, bar_y),
            arrowprops=dict(arrowstyle="-", color="#343434", lw=1.2),
        )
        pval_label = f"p={pvalue:.4f} {stars}" if pvalue >= 0.0001 else f"p<0.0001 {stars}"
        ax.text(1.5, bar_y + y_range * 0.015, pval_label,
                ha="center", va="bottom", fontsize=8.5, color="#343434", fontweight="500")

    ax.set_xticks(positions)
    ax.set_xticklabels([label_a, label_b], rotation=20, ha="right")
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
# Strata (split by fitness function when multiple are present)
# ---------------------------------------------------------------------------

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
) -> str:
    if label_a is not None and label_b is not None:
        base = (
            f"{param_xlabel_es(param)}: {label_a} vs {label_b}"
            f" — {metric_ylabel_es(metric)}"
        )
    else:
        base = (
            f"Comparación por {param_xlabel_es(param).lower()}"
            f" — {metric_ylabel_es(metric)}"
        )
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
            if len(labels_all) < 2:
                continue

            for metric in metrics:
                data, labels = collect_groups(stratum_rows, param, metric)
                if len(labels) < 2 or all(len(d) == 0 for d in data):
                    continue

                xlabel = param_xlabel_es(param)
                ylabel = metric_ylabel_es(metric)

                # --- All-methods plot ---
                title = build_title(param, metric, stratum_value, multi_fitness)
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
        required=True,
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
        "--list-columns",
        action="store_true",
        help="Print plottable column names from the CSV header and exit.",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.list_columns:
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
    )

    if paths:
        print(f"\nTotal: {len(paths)} figure(s) written to {out_dir}")
    else:
        print("No figures written (check columns and that parameters vary across rows).")


if __name__ == "__main__":
    main()
