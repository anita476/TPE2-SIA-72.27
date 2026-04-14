# TPE2-SIA-72.27

## Requirements

- Python 3.10+
- Dependencies:

```bash
pip install -r requirements.txt
```

## Optional: Virtual Environment

To run the project inside a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Single Run (`main.py`)

### CLI example

```
python main.py --input_image input/france-flag.png --output_image output/result.png --triangles 200 --population-size 100 --generations 5000 --k 60 --selector elite --crossover uniform --mutation multigen_limited --mutation-rate 0.20 --mutation-strength 0.20 --fitness rmse5 --survival_strategy exclusive
```

### Config-driven run (recommended)

`main.py` accepts flat JSON configs (all keys at the top level, like `configs/france_quick.json`):

```bash
python main.py --config configs/france_quick.json
```

CLI arguments override config values:

```bash
python main.py --config configs/france_quick.json --selector boltzmann --generations 7000
```

> **Note:** Grid configs with `base`/`grid` keys (e.g. `configs/low_improved.json`) are only for `experiment_runner.py` — they will not work with `main.py`.

## Main Parameters (current defaults)

### Core

| Argument              | Default      | Description                                                   |
| --------------------- | ------------ | ------------------------------------------------------------- |
| `--input_image`       | required     | Input image path                                              |
| `--output_image`      | `output.png` | Output image path                                             |
| `--triangles`         | required     | Triangles per individual                                      |
| `--population-size`   | required     | Population size                                               |
| `--generations`       | required     | Maximum generations                                           |
| `--k`                 | required     | Parents selected and offspring generated per generation       |
| `--mutation-rate`     | `0.1`        | Mutation probability (depends on mutation operator semantics) |
| `--mutation-strength` | `0.3`        | Mutation intensity (gene perturbation magnitude)              |
| `--seed`              | `None`       | Random seed for reproducibility                               |
| `--snapshot-interval` | `0`          | Save snapshot every N generations (`0` disables)              |
| `--output-dir`        | `snapshots`  | Snapshot directory                                            |
| `--tournament-size`   | `2`          | Tournament size (M) for deterministic tournament              |

### Selection (`--selector`)

| Value              | Default | Description                                              |
| ------------------ | ------- | -------------------------------------------------------- |
| `elite`            | yes     | Deterministic rank-based repetition of top individuals   |
| `roulette`         | no      | Fitness-proportionate random sampling                    |
| `universal`        | no      | SUS variant of roulette with lower sampling variance     |
| `ranking`          | no      | Linear rank weighting + stochastic sampling              |
| `boltzmann`        | no      | Temperature-based probabilistic selection                |
| `tournament_det`   | no      | Tournament with deterministic winner                     |
| `tournament_stoch` | no      | Tournament where winner depends on threshold probability |

**Tournament-specific:**

| Argument                | Default | Description                                              |
| ----------------------- | ------- | -------------------------------------------------------- |
| `--tournament-threshold` | `0.75`  | Win probability for better individual in `tournament_stoch` (0.5–1.0) |

### Crossover (`--crossover`)

| Value       | Default | Description                        |
| ----------- | ------- | ---------------------------------- |
| `annular`   | no      | Ring-like segment exchange         |
| `one_point` | no      | Single cut-point crossover         |
| `swapper`   | no      | Swap-based crossover               |
| `two_point` | yes     | Two cut-point crossover            |
| `uniform`   | no      | Gene-wise mixing using random mask |

### Mutation (`--mutation`)

| Value              | Default | Description                                             |
| ------------------ | ------- | ------------------------------------------------------- |
| `gen`              | no      | Mutates one gene when mutation triggers                 |
| `multigen_limited` | no      | Mutates a random subset of genes when mutation triggers |
| `multigen_uniform` | yes     | Each gene mutates independently with `mutation_rate`    |
| `complete`         | no      | Mutates all genes when mutation triggers                |

### Fitness (`--fitness`)

| Value   | Default | Description                           |
| ------- | ------- | ------------------------------------- |
| `mae`   | yes     | Mean Absolute Error-based fitness     |
| `mse`   | no      | Mean Squared Error-based fitness      |
| `rmse`  | no      | Root Mean Squared Error-based fitness |
| `mse5`  | no      | Weighted MSE variant                  |
| `rmse5` | no      | Weighted RMSE variant                 |

Fitness semantics in this project: **higher is better**.

### Survival (`--survival_strategy`)

| Value       | Default | Description                                           |
| ----------- | ------- | ----------------------------------------------------- |
| `additive`  | yes     | Parent + offspring pool, then select next population  |
| `exclusive` | no      | Prefer offspring and fill from parents only if needed |

### Boltzmann-specific

Used only when `--selector boltzmann`:

| Argument              | Default  | Description                                                 |
| --------------------- | -------- | ----------------------------------------------------------- |
| `--temperature`       | `1.0`    | Initial temperature \(T0\)                                  |
| `--temperature-min`   | `0.1`    | Minimum temperature \(Tc\)                                  |
| `--temperature-decay` | `-0.001` | Exponential decay `k` in `T(g) = Tc + (T0 - Tc) * exp(k*g)` |

### Early stopping

Any combination can be used simultaneously (`--generations` stays as hard cap):

| Argument               | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| `--target-fitness`     | Stop when best fitness is at least this value                            |
| `--convergence-window` | Stop when best fitness does not improve enough for this many generations |
| `--convergence-delta`  | Improvement threshold used by convergence window (`1e-4` default)        |
| `--time-limit`         | Stop after this many seconds                                             |

---

## Experiment Grid (`analysis/experiment_runner.py`)

Runs hyperparameter combinations from a JSON config and writes:

- `results.csv`
- `results_history.csv` (same base name + `_history`)

### Basic

```bash
python analysis/experiment_runner.py --config configs/experiment_grid.json --output output/experiments/results.csv
```

### Parallel

```bash
python analysis/experiment_runner.py --config configs/experiment_grid.json --output output/experiments/results.csv --workers 6
```

Use all CPU cores:

```bash
python analysis/experiment_runner.py --config configs/experiment_grid.json --output output/experiments/results.csv --workers 0
```

Save final image per combination:

```bash
python analysis/experiment_runner.py --config configs/experiment_grid.json --output output/experiments/results.csv --save-images
```

### Config format (`base` + `grid` + optional `pairs`)

```json
{
  "base": {
    "input_image": "input/france-flag.png",
    "triangles": 200,
    "population_size": 100,
    "generations": 5000,
    "k": 60,
    "selector": "elite",
    "crossover": "uniform",
    "mutation": "multigen_limited",
    "mutation_rate": 0.2,
    "mutation_strength": 0.2,
    "survival_strategy": "exclusive",
    "fitness": "rmse5"
  },
  "grid": {
    "selector": ["elite", "ranking", "tournament_stoch"],
    "seed": [1, 2, 3, 4, 5]
  },
  "pairs": [
    {
      "mutation_rate": [0.15, 0.2, 0.28, 0.35],
      "mutation_strength": [0.15, 0.2, 0.28, 0.35]
    }
  ]
}
```

- `grid`: Cartesian product across keys.
- `pairs`: zipped values inside each pair block, then combined with grid.

### CLI arguments

| Argument        | Default       | Description                            |
| --------------- | ------------- | -------------------------------------- |
| `--config`      | required      | Path to experiment grid JSON           |
| `--output`      | `results.csv` | Output CSV path                        |
| `--save-images` | disabled      | Save final image per combination       |
| `--workers`     | `1`           | Parallel workers (`0` = all CPU cores) |

---

## Comparative Plots (`plotters/plot_experiment_comparisons.py`)

### CLI arguments

| Argument             | Default                            | Description                                            |
| -------------------- | ---------------------------------- | ------------------------------------------------------ |
| `--csv`              | required in box/scatter mode       | Main results CSV                                       |
| `--history-csv`      | optional                           | History CSV for convergence mode                       |
| `--out`              | required (except `--list-columns`) | Output directory                                       |
| `--x-axis`           | all hyperparameter columns         | Category column(s)                                     |
| `--y-axis`           | all available metrics              | Metric column(s) to plot                               |
| `--selectors`        | all                                | Filter by selector names                               |
| `--filter KEY=VALUE` | none                               | Generic row filtering                                  |
| `--pairwise`         | disabled                           | Pairwise boxplots between categories                   |
| `--scatter`          | disabled                           | Efficiency scatter plots                               |
| `--convergence-x`    | `generation`                       | Convergence X axis (`generation` or `elapsed_seconds`) |
| `--show-seeds`       | disabled                           | Show thin per-seed curves in convergence               |
| `--show-mean`        | disabled                           | Add mean markers on boxplots                           |
| `--add-stats`        | disabled                           | Add statistical text annotations                       |
| `--list-columns`     | disabled                           | Print plottable columns and exit                       |

### List available columns

```bash
python plotters/plot_experiment_comparisons.py --csv output/test1/results.csv --list-columns
```

### Boxplots for selected metrics

```bash
python plotters/plot_experiment_comparisons.py --csv output/test1/results.csv --out output/test1/plots --x-axis selector --y-axis best_fitness --y-axis elapsed_seconds
```

### Pairwise comparisons

```bash
python plotters/plot_experiment_comparisons.py --csv output/test1/results.csv --out output/test1/plots --x-axis selector --y-axis best_fitness --pairwise
```

### Scatter efficiency plots

```bash
python plotters/plot_experiment_comparisons.py --csv output/test1/results.csv --out output/test1/plots --x-axis selector --scatter
```

### Convergence by generation

```bash
python plotters/plot_experiment_comparisons.py --history-csv output/test1/results_history.csv --out output/test1/plots --x-axis selector --convergence-x generation
```

### Convergence by elapsed time

```bash
python plotters/plot_experiment_comparisons.py --history-csv output/test1/results_history.csv --out output/test1/plots --x-axis selector --convergence-x elapsed_seconds
```

Optional: show individual seed lines in convergence plots with `--show-seeds`.

---

## Mutation × Selection Grid (`analysis/mutation_selection_grid_experiment.py`)

### CLI arguments

| Argument               | Default                          | Description                             |
| ---------------------- | -------------------------------- | --------------------------------------- |
| `--config`             | optional                         | Config JSON (recommended)               |
| `--input-image`        | required if not in config        | Input image path                        |
| `--num-runs`           | `5`                              | Runs per mutation-selection combination |
| `--triangles`          | `200`                            | Number of triangles                     |
| `--population-size`    | `100`                            | Population size                         |
| `--generations`        | `500`                            | Max generations                         |
| `--k`                  | `40`                             | Offspring size                          |
| `--crossover`          | `annular`                        | Crossover operator                      |
| `--fitness`            | `rmse5`                          | Fitness function                        |
| `--survival-strategy`  | `exclusive`                      | Survival strategy                       |
| `--mutation-rate`      | `0.22`                           | Mutation rate                           |
| `--mutation-strength`  | `0.28`                           | Mutation strength                       |
| `--convergence-window` | `20`                             | Early-stop convergence window           |
| `--output`             | `output/mutation_selection_grid` | Output directory                        |
| `--max-workers`        | `None`                           | Parallel workers (`None` = sequential)  |

Run from config:

```bash
python analysis/mutation_selection_grid_experiment.py --config configs/mutation_selection_grid_uniform.json
```

Outputs `results.json` inside the configured output directory.

### Heatmap (`plotters/plot_mutation_selection_heatmap.py`)

```bash
python plotters/plot_mutation_selection_heatmap.py --results output/mutation_selection_grid_uniform/results.json --output output/mutation_selection_grid_uniform/mutation_selection_heatmap_avg.png
```

Plot standard deviation instead of average:

```bash
python plotters/plot_mutation_selection_heatmap.py --results output/mutation_selection_grid_uniform/results.json --output output/mutation_selection_grid_uniform/mutation_selection_heatmap_std.png --metric std
```

---

## Tournament Threshold Analysis (`analysis/threshold_analyzer.py`)

### CLI arguments

| Argument               | Default                     | Description                       |
| ---------------------- | --------------------------- | --------------------------------- |
| `--config`             | optional                    | Config JSON (recommended)         |
| `--input-image`        | required if not in config   | Input image path                  |
| `--num-runs`           | `5`                         | Runs per threshold                |
| `--num-thresholds`     | `10`                        | Number of threshold values tested |
| `--triangles`          | `200`                       | Number of triangles               |
| `--population-size`    | `100`                       | Population size                   |
| `--generations`        | `1000`                      | Max generations                   |
| `--k`                  | `40`                        | Offspring size                    |
| `--crossover`          | `annular`                   | Crossover operator                |
| `--fitness`            | `rmse5`                     | Fitness function                  |
| `--survival-strategy`  | `exclusive`                 | Survival strategy                 |
| `--mutation`           | `multigen_limited`          | Mutation operator                 |
| `--mutation-rate`      | `0.22`                      | Mutation rate                     |
| `--mutation-strength`  | `0.28`                      | Mutation strength                 |
| `--convergence-window` | disabled                    | No-improvement window             |
| `--convergence-delta`  | `1e-4`                      | Minimum improvement threshold     |
| `--output-dir`         | `output/threshold_analysis` | Output directory                  |
| `--workers`            | `1`                         | Parallel workers                  |
| `--verbose`            | disabled                    | Print generation progress         |
| `--seed`               | `None`                      | Base random seed                  |

Run via config:

```bash
python analysis/threshold_analyzer.py --config configs/threshold_france.json
```

Or via CLI:

```bash
python analysis/threshold_analyzer.py --input-image input/france-flag.png --num-runs 5 --num-thresholds 10 --workers 6
```

This script generates summary JSON plus threshold-related plots in its output directory.

---

## Additional Plotters

### Threshold Overlay (`plotters/plot_threshold_overlay.py`)

Overlay multiple threshold-analysis `results.json` files on the same figure.

Example:

```bash
python plotters/plot_threshold_overlay.py --results output/threshold_analysis/results.json --results output/threshold_analysis_france_v2/results.json --auto-names dir --output output/threshold_analysis/overlay.png
```

| Argument        | Default               | Description                               |
| --------------- | --------------------- | ----------------------------------------- |
| `--results`     | required (repeatable) | Path(s) to threshold `results.json` files |
| `--input-names` | none                  | Custom curve labels (space-separated)     |
| `--auto-names`  | `dir`                 | Auto-label mode: `dir`, `file`, or `path` |
| `--output`      | interactive display   | Output image path                         |
| `--metric`      | `avg_best_fitness`    | Metric key from results JSON              |
| `--no-std`      | disabled              | Hide std-dev shaded bands                 |

### Tournament M Results Plotter (`plotters/plot_tournament_m_results.py`)

Plots outputs from tournament-size-vs-triangle-count experiments.

Example:

```bash
python plotters/plot_tournament_m_results.py --summary output/tournament_m_triangles/summary.json --results-dir output/tournament_m_triangles --output output/tournament_m_triangles/plots
```

| Argument        | Default                        | Description                                         |
| --------------- | ------------------------------ | --------------------------------------------------- |
| `--summary`     | required                       | `summary.json` generated by tournament-M experiment |
| `--results-dir` | summary parent directory       | Directory with `results_triangles_N.json` files     |
| `--output`      | `plots/tournament_m_triangles` | Output directory for generated plots                |

### Generic CSV Plotter (`plotters/plotter.py`)

Utility plotter for quick CSV inspections.

Examples:

```bash
python plotters/plotter.py output/test1/results.csv --columns
python plotters/plotter.py output/test1/results.csv --x selector --y best_fitness -o output/test1/plots/quick_best_fitness.png
```

| Argument         | Default             | Description                                      |
| ---------------- | ------------------- | ------------------------------------------------ |
| `file`           | required            | Input CSV file                                   |
| `--x`            | none                | X-axis column                                    |
| `--y`            | none                | Y-axis column                                    |
| `--columns`      | disabled            | List available columns and exit                  |
| `--output`, `-o` | interactive display | Output image path                                |
| `--tight`        | disabled            | Tight Y-range to emphasize differences           |
| `--logx`         | disabled            | Log scale on X-axis                              |
| `--logy`         | disabled            | Log scale on Y-axis                              |
| `--history`      | none                | History-column mode (line plot grouped by `--x`) |
