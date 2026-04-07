# TPE2-SIA-72.27
Algoritmos Genéticos. Segundo trabajo práctico para Sistemas de Inteligencia Artificial.

## Setup
...

## Usage

### Via CLI

```bash
python3 main.py \
  --input_image input/Flag_of_France.svg.png \
  --output_image output/result.png \
  --triangles 200 \
  --population-size 100 \
  --generations 500 \
  --k 30
```

### Via config file (recommended)

```bash
python3 main.py --config configs/france_quick.json
```

CLI args override config values:

```bash
python3 main.py --config configs/france_quick.json --generations 1000 --selector boltzmann
```

Config files use underscores for multi-word keys (`population_size`, `snapshot_interval`). See `configs/france_quick.json` for an example.

---

## Parameters

### Core

| Argument | Default | Description |
|---|---|---|
| `--input_image` | required | Path to input image |
| `--output_image` | `output.png` | Path to output image |
| `--triangles` | required | Number of triangles per individual |
| `--population-size` | required | Number of individuals in the population |
| `--generations` | required | Maximum number of generations |
| `--k` | required | Parents selected and offspring produced per generation |
| `--mutation-rate` | `0.1` | Probability of mutating each triangle |
| `--mutation-strength` | `0.3` | Magnitude of each mutation |
| `--snapshot-interval` | `0` | Save image snapshot every N generations (0 = disabled) |
| `--output-dir` | `snapshots` | Directory for snapshots |

### Selection (`--selector`)

| Value | Description |
|---|---|
| `elite` (default) | Repeats best individuals proportionally by rank |
| `roulette` | Fitness-proportionate selection (`1/fitness` as weight) |
| `universal` | Stochastic Universal Sampling - same weights as roulette, lower variance |
| `ranking` | Linear ranking: weight = `(N - rank(i)) / N` |
| `boltzmann` | Annealed selection - see Boltzmann section below |

### Crossover (`--crossover`)

| Value | Description |
|---|---|
| `two_point` (default) | Swaps middle segment between two random cut points |
| `one_point` | Swaps tail after a single random cut point |

### Fitness (`--fitness`)

| Value | Description |
|---|---|
| `mae` (default) | Mean Absolute Error over RGBA channels |
| `mse` | Mean Squared Error — penalises large errors more |
| `rmse` | Root Mean Squared Error |

All functions: **lower = better**.

### Survival strategy (`--survival_strategy`)

| Value | Description |
|---|---|
| `additive` (default) | Pool = current population + offspring; select top K |
| `exclusive` | Fill from offspring first; top up from parents only if needed |

---

## Boltzmann Selection

Uses an exponential temperature schedule:
*#TODO*: ask if we need to consider adding more or trying out different schedules

```
T(t) = T_c + (T_0 - T_c) * e^(k*t)
```

| Argument | Default | Description |
|---|---|---|
| `--temperature` | `50.0` | T_0: initial temperature |
| `--temperature-min` | `1.0` | T_c: convergence floor |
| `--temperature-decay` | `-0.005` | k: decay rate, must be negative |

High temperature -> uniform selection (exploration). Low temperature -> best individual dominates (exploitation). To be ~99% cooled by generation G: `k ≈ -4.6 / G`.

---

## Early Stopping

Any combination of the following can be used together: stops on whichever triggers first. `--generations` remains the hard cap.

| Argument | Description |
|---|---|
| `--target-fitness` | Stop when best fitness <= this value |
| `--convergence-window` | Stop when best fitness hasn't improved by `--convergence-delta` over this many generations |
| `--convergence-delta` | Minimum improvement threshold (default `1e-4`) |
| `--time-limit` | Stop after this many seconds |

```bash
# stop when MAE <= 5, or no improvement over 100 gens, or after 5 minutes
python3 main.py --config configs/france_quick.json \
  --target-fitness 5.0 --convergence-window 100 --time-limit 300
```

---

## Grid Search / Experiments

Run all combinations of selected hyperparameters and save results to CSV:

```bash
python3 experiment_runner.py --config configs/experiment_grid.json --output results.csv

# also save output image per combination
python3 experiment_runner.py --config configs/experiment_grid.json --output results.csv --save-images
```

Grid config format:

```json
{
  "base": {
    "input_image": "input/Flag_of_France.svg.png",
    "triangles": 200,
    "mutation_rate": 0.1,
    "mutation_strength": 0.3
  },
  "grid": {
    "selector":          ["elite", "roulette", "boltzmann"],
    "crossover":         ["one_point", "two_point"],
    "population_size":   [50, 100],
    "generations":       [500],
    "k":                 [20, 50]
  }
}
```

`base` values apply to every run. `grid` values are crossed — every combination is tested. Results are written to CSV incrementally (safe to interrupt) with columns for each grid param plus `best_fitness` and `generations_run`. See `configs/experiment_grid.json` for a full example.
