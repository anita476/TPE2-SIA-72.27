from __future__ import annotations

from crossovers.annular import annular_crossover as annular
from crossovers.one_point import one_point_crossover as one_point
from crossovers.two_point import two_point_crossover as two_point
from crossovers.uniform import uniform_crossover as uniform
from fitness.mae import mae_fitness
from fitness.mse import mse_fitness
from fitness.mse5 import mse5_fitness
from fitness.rmse import rmse_fitness
from fitness.rmse5 import rmse5_fitness
from mutations.complete import complete_mutation
from mutations.gen import gen_mutation
from mutations.multigen_limited import multigen_limited_mutation
from mutations.multigen_uniform import multigen_uniform_mutation
from selection.boltzmann import AnnealedBoltzmann
from selection.elite import elite_selection as elite
from selection.ranking import ranking
from selection.roulette import roulette_selection as roulette
from selection.tournament_deterministic import tournament_deterministic
from selection.tournament_stochastic import tournament_stochastic
from selection.universal import universal_selection as universal
from survival_strategies.additive import additive_survival as additive
from survival_strategies.exclusive import exclusive_survival as exclusive
from utils.stop_conditions import StopCondition, any_of, no_improvement, target_fitness, time_limit

CROSSOVER_MAP = {
    "annular": annular,
    "one_point": one_point,
    "two_point": two_point,
    "uniform": uniform,
}

FITNESS_MAP = {
    "mae":  mae_fitness,
    "mse":  mse_fitness,
    "rmse": rmse_fitness,
    "mse5": mse5_fitness,
    "rmse5": rmse5_fitness,
}

MUTATION_MAP = {
    "gen":               gen_mutation,
    "multigen_limited":  multigen_limited_mutation,
    "multigen_uniform":  multigen_uniform_mutation,
    "complete":          complete_mutation,
}

SURVIVAL_MAP = {
    "additive":  additive,
    "exclusive": exclusive,
}


def build_selector(name: str, temperature: float, temperature_min: float, temperature_decay: float):
    selector_map = {
        "elite":            elite,
        "roulette":         roulette,
        "universal":        universal,
        "ranking":          ranking,
        "boltzmann":        AnnealedBoltzmann(temperature, temperature_min, temperature_decay),
        "tournament_det":   tournament_deterministic,
        "tournament_stoch": tournament_stochastic,
    }
    return selector_map[name]


def build_stop_condition(
    target_fitness_val: float | None,
    convergence_window: int | None,
    convergence_delta: float,
    time_limit_secs: float | None,
) -> StopCondition | None:
    conditions = []
    if target_fitness_val is not None:
        conditions.append(target_fitness(target_fitness_val))
    if convergence_window is not None:
        conditions.append(no_improvement(convergence_window, convergence_delta))
    if time_limit_secs is not None:
        conditions.append(time_limit(time_limit_secs))
    return any_of(*conditions) if conditions else None
