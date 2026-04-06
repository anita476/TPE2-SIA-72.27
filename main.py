import sys
from io import BytesIO

from PIL import Image

from crossovers.one_point import one_point_crossover as one_point
from crossovers.two_point import two_point_crossover as two_point
from fitness.mae import mae_fitness
from fitness.mse import mse_fitness
from fitness.rmse import rmse_fitness
from genetic_algorithm import run_genetic_algorithm
from input_output_handler import read_image, save_image, parse_arguments
from mutations.gen import gen_mutation
from mutations.multigen_limited import multigen_limited_mutation
from mutations.multigen_uniform import multigen_uniform_mutation
from mutations.complete import complete_mutation
from selection.boltzmann import AnnealedBoltzmann
from selection.elite import elite_selection as elite
from selection.ranking import ranking
from selection.roulette import roulette_selection as roulette
from selection.universal import universal_selection as universal
from survival_strategies.additive import additive_survival as additive
from survival_strategies.exclusive import exclusive_survival as exclusive

CROSSOVER_MAP = {
    "one_point": one_point,
    "two_point": two_point,
}

FITNESS_MAP = {
    "mae": mae_fitness,
    "mse": mse_fitness,
    "rmse": rmse_fitness,
}

MUTATION_MAP = {
    "gen": gen_mutation,
    "multigen_limited": multigen_limited_mutation,
    "multigen_uniform": multigen_uniform_mutation,
    "complete": complete_mutation,
}

SURVIVAL_MAP = {
    "additive": additive,
    "exclusive": exclusive,
}


def main():
    args = parse_arguments()

    try:
        image_bytes = read_image(args.input_image)
    except (FileNotFoundError, OSError) as e:
        print(f"Error reading image: {e}")
        sys.exit(1)

    with Image.open(BytesIO(image_bytes)) as img:
        source_image = img.convert("RGBA").copy()

    selector_map = {
        "elite":      elite,
        "roulette":   roulette,
        "universal":  universal,
        "ranking":    ranking,
        "boltzmann":  AnnealedBoltzmann(args.temperature, args.temperature_min, args.temperature_decay),
    }

    result = run_genetic_algorithm(
        source_image=source_image,
        num_triangles=args.triangles,
        population_size=args.population_size,
        generations=args.generations,
        k=args.k,
        selector=selector_map[args.selector],
        crossover=CROSSOVER_MAP[args.crossover],
        fitness_fn=FITNESS_MAP[args.fitness],
        survival_strategy=SURVIVAL_MAP[args.survival_strategy],
        mutation_fn=MUTATION_MAP[args.mutation],
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        snapshot_interval=args.snapshot_interval,
        output_dir=args.output_dir,
    )

    try:
        save_image(result, args.output_image)
    except OSError as e:
        print(f"Error saving image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
