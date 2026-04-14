import argparse
import ast
import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


STYLE = {
    "figure_bg": "#fff5ec",
    "axes_bg": "#fff5ec",
    "text_axis": "#343434",
    "grid": "#e8dcd0",
    "grid_minor": "#d4c8bc",
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

SERIES_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
]
BAR_FACE = "#4a90d9"
BAR_EDGE = "#2c6cb0"
POINT_FACE = "#e67e22"
POINT_EDGE = "#c45f1a"
ERROR_COLOR = "#343434"


Y_LABELS_ES = {
    "best_fitness": "Mejor fitness",
    "initial_best_fitness": "Mejor fitness inicial",
    "final_mean_fitness": "Fitness media final",
    "final_std_fitness": "Desvío estándar del fitness final",
    "final_population_diversity": "Diversidad final de la población",
    "improvement_absolute": "Mejora absoluta",
    "improvement_ratio": "Razón de mejora",
    "improvement_percent": "Mejora porcentual (%)",
    "generations_run": "Generaciones ejecutadas",
    "elapsed_seconds": "Tiempo transcurrido (s)",
    "best_fitness_history": "Mejor fitness",
    "mutation_rate": "Tasa de mutación",
    "mutation_strength": "Fuerza de mutación",
    "population_size": "Tamaño de población",
    "triangles": "Cantidad de triángulos",
    "tournament_threshold": "Umbral del torneo",
}


def es_ylabel(col: str) -> str:
    return Y_LABELS_ES.get(col, col)


def setup_axes_style(fig, ax):
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


def _axis_decimals(ax):
    ax.figure.canvas.draw()
    labels = [t.get_text() for t in ax.yaxis.get_ticklabels() if t.get_text()]
    for label in labels:
        if "." in label:
            return len(label.split(".")[-1])
    return 2


def add_value_labels(ax, xs, ys, fmt=None, tops=None):
    if fmt is None:
        fmt = f".{_axis_decimals(ax)}f"
    if tops is None:
        tops = ys
    for x, y, top in zip(xs, ys, tops):
        ax.annotate(
            f"{y:{fmt}}",
            (x, top),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="top",
            fontsize=8,
            color=STYLE["text_axis"],
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": STYLE["grid"],
                "linewidth": 0.6,
                "alpha": 0.9,
            },
        )


def _plain_log_tick(value, _pos):
    if value <= 0:
        return ""
    if value < 10000:
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value))}"
        return f"{value:g}"
    return f"{value:.0e}"


def configure_log_axis_ticks(ax, axis_name, values):
    axis = ax.xaxis if axis_name == "x" else ax.yaxis
    axis.set_major_locator(ticker.FixedLocator(values))
    axis.set_major_formatter(ticker.FuncFormatter(_plain_log_tick))
    axis.set_minor_locator(ticker.NullLocator())


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def resolve_column(rows, col):
    """Return list of values for *col*. Tries numeric conversion first."""
    raw = [r[col] for r in rows]
    try:
        return [float(v) for v in raw], True
    except ValueError:
        return raw, False


def plot_history(rows, args):
    if args.x is None:
        print("--history requiere --x (columna de agrupamiento, ej: crossover).")
        sys.exit(1)

    groups = defaultdict(list)
    for r in rows:
        series = ast.literal_eval(r[args.history])
        groups[r[args.x]].append(np.asarray(series, dtype=float))

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, 5.5))
    setup_axes_style(fig, ax)

    for idx, key in enumerate(groups):
        runs = groups[key]
        min_len = min(len(a) for a in runs)
        stack = np.stack([a[:min_len] for a in runs])
        mean = stack.mean(axis=0)
        gens = np.arange(min_len)
        ax.plot(
            gens,
            mean,
            label=key,
            color=SERIES_COLORS[idx % len(SERIES_COLORS)],
            linewidth=2.0,
        )

    ax.set_xlabel("Generación", color=STYLE["text_axis"])
    ax.set_ylabel(es_ylabel(args.history), color=STYLE["text_axis"])
    legend = ax.legend(title=args.x, framealpha=0.8, edgecolor=STYLE["grid"], facecolor="white")
    legend.get_title().set_color(STYLE["text_axis"])
    for text in legend.get_texts():
        text.set_color(STYLE["text_axis"])
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
        print(f"Guardado en {args.output}")
    else:
        plt.show()


def plot(args):
    rows = load_csv(args.file)

    if args.columns:
        print("Columnas disponibles:")
        for c in rows[0].keys():
            print(f"  - {c}")
        sys.exit(0)

    if args.history:
        plot_history(rows, args)
        return

    if args.x is None or args.y is None:
        print("Se requieren --x y --y. Usa --columns para ver las disponibles.")
        sys.exit(1)

    x_vals, x_numeric = resolve_column(rows, args.x)
    y_vals, _ = resolve_column(rows, args.y)

    # Group by x value
    groups = defaultdict(list)
    for xv, yv in zip(x_vals, y_vals):
        groups[xv].append(yv)

    needs_agg = any(len(v) > 1 for v in groups.values())

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(8, 5))
    setup_axes_style(fig, ax)

    if needs_agg:
        keys = sorted(groups.keys()) if x_numeric else list(groups.keys())
        means = [np.mean(groups[k]) for k in keys]
        stds = [np.std(groups[k]) for k in keys]

        if x_numeric:
            ax.errorbar(
                keys,
                means,
                yerr=stds,
                fmt="o-",
                color=BAR_FACE,
                ecolor=ERROR_COLOR,
                markerfacecolor=POINT_FACE,
                markeredgecolor=POINT_EDGE,
                linewidth=1.8,
                markersize=5,
                capsize=4,
                capthick=1.2,
            )
            label_xs, label_ys = keys, means
            label_tops = means
        else:
            x_pos = range(len(keys))
            ax.bar(
                x_pos,
                means,
                yerr=stds,
                capsize=4,
                alpha=0.85,
                color=BAR_FACE,
                edgecolor=BAR_EDGE,
                linewidth=1.0,
                ecolor=ERROR_COLOR,
            )
            label_xs, label_ys = list(x_pos), means
            label_tops = means
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(keys, rotation=30, ha="right")
    else:
        keys = sorted(groups.keys()) if x_numeric else list(groups.keys())
        vals = [groups[k][0] for k in keys]

        if x_numeric:
            ax.plot(
                keys,
                vals,
                "o-",
                color=BAR_FACE,
                markerfacecolor=POINT_FACE,
                markeredgecolor=POINT_EDGE,
                linewidth=1.8,
                markersize=5,
            )
            label_xs, label_ys, label_tops = keys, vals, None
        else:
            x_pos = range(len(keys))
            ax.bar(
                x_pos,
                vals,
                alpha=0.85,
                color=BAR_FACE,
                edgecolor=BAR_EDGE,
                linewidth=1.0,
            )
            label_xs, label_ys, label_tops = list(x_pos), vals, None
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(keys, rotation=30, ha="right")

    ax.set_xlabel(es_ylabel(args.x), color=STYLE["text_axis"])
    ax.set_ylabel(es_ylabel(args.y), color=STYLE["text_axis"])

    if args.logx:
        ax.set_xscale("log")
        if x_numeric:
            configure_log_axis_ticks(ax, "x", keys)
    if args.logy:
        ax.set_yscale("log")

    if args.tight and needs_agg:
        lows = [m - s for m, s in zip(means, stds)]
        highs = [m + s for m, s in zip(means, stds)]
        lo, hi = min(lows), max(highs)
        span = hi - lo if hi > lo else max(abs(hi), 1.0) * 0.1
        pad = span * 0.25
        ax.set_ylim(lo - pad, hi + pad)

    label_fmt = ".3f" if needs_agg else None
    add_value_labels(ax, label_xs, label_ys, fmt=label_fmt, tops=label_tops)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
        print(f"Guardado en {args.output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Graficador de CSV")
    parser.add_argument("file", help="Archivo CSV de entrada")
    parser.add_argument("--x", help="Columna para el eje X")
    parser.add_argument("--y", help="Columna para el eje Y")
    parser.add_argument("--columns", action="store_true",
                        help="Listar columnas disponibles y salir")
    parser.add_argument("--output", "-o", help="Guardar imagen en archivo (ej: plot.png)")
    parser.add_argument("--tight", action="store_true",
                        help="Ajustar eje Y al rango de datos para resaltar diferencias")
    parser.add_argument("--logx", action="store_true",
                        help="Usar escala logaritmica en el eje X")
    parser.add_argument("--logy", action="store_true",
                        help="Usar escala logaritmica en el eje Y")
    parser.add_argument("--history",
                        help="Columna con lista por generación (ej: best_fitness_history); grafica una lí­nea por valor de --x con banda ±std")
    plot(parser.parse_args())


if __name__ == "__main__":
    main()
