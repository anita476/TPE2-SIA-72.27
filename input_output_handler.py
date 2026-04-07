import argparse
import json
import os


def read_image(path: str) -> bytes:
    with open(path, "rb") as f:
        image = f.read()
        print(f'Image read: "{os.path.basename(path)}", {len(image)} bytes')
        return image


def save_image(image: bytes, path: str) -> None:
    with open(path, "wb") as f:
        f.write(image)
        print(f'Image saved: "{os.path.basename(path)}", {len(image)} bytes')


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Genetic algorithm that approximates an image with semi-transparent triangles."
    )

    argument_parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file. CLI args override config values.")

    # required args become optional when a config file is provided
    argument_parser.add_argument("--input_image", type=str, default=None, help="Path to the input image")
    argument_parser.add_argument("--output_image", type=str, default="output.png", help="Path to the output image")
    argument_parser.add_argument("--triangles", type=int, default=None, help="Triangles to use in the approximation")
    argument_parser.add_argument("--population-size", type=int, default=None, help="Number of individuals in the population")
    argument_parser.add_argument("--generations", type=int, default=None, help="Number of generations to run")
    argument_parser.add_argument("--k", type=int, default=None, help="Number of selected parents and offspring generated per generation")
    argument_parser.add_argument("--mutation-rate", type=float, default=0.1, help="Probability of mutating each triangle (0.0 to 1.0)")
    argument_parser.add_argument("--mutation-strength", type=float, default=0.3, help="Magnitude of each mutation (0.0 to 1.0)")
    argument_parser.add_argument("--snapshot-interval", type=int, default=0, help="Save snapshot every N generations (0=disabled)")
    argument_parser.add_argument("--output-dir", type=str, default="snapshots", help="Directory for snapshot images")
    argument_parser.add_argument("--selector", type=str, default="elite", choices=["elite", "roulette", "universal", "ranking", "boltzmann", "tournament_det", "tournament_stoch"], help="Selection method")
    argument_parser.add_argument("--crossover", type=str, default="two_point", choices=["one_point", "two_point"], help="Crossover operator")
    argument_parser.add_argument("--fitness", type=str, default="mae", choices=["mae", "mse", "rmse"], help="Fitness function; lower is better")
    argument_parser.add_argument("--mutation", type=str, default="multigen_uniform", choices=["gen", "multigen_limited", "multigen_uniform", "complete"], help="Mutation strategy: gen (single gene), multigen_limited (random subset), multigen_uniform (each gene independently), complete (all genes)")
    argument_parser.add_argument("--survival_strategy", type=str, default="additive", choices=["additive", "exclusive"], help="Survival strategy for the next generation")
    argument_parser.add_argument("--temperature", type=float, default=50.0, help="Initial temperature T_0 for Boltzmann selection")
    argument_parser.add_argument("--temperature-min", type=float, default=1.0, help="Convergence temperature T_c for Boltzmann selection")
    argument_parser.add_argument("--temperature-decay", type=float, default=-0.005, help="Decay constant k for Boltzmann schedule: T(t) = T_c + (T_0 - T_c) * e^(k*t), k < 0 for cooling")
    argument_parser.add_argument("--target-fitness", type=float, default=None, help="Stop when best fitness reaches this value")
    argument_parser.add_argument("--convergence-window", type=int, default=None, help="Stop when best fitness hasn't improved by --convergence-delta over this many generations")
    argument_parser.add_argument("--convergence-delta", type=float, default=1e-4, help="Minimum improvement required within --convergence-window")
    argument_parser.add_argument("--time-limit", type=float, default=None, help="Stop after this many seconds")

    # parse once to get --config, then apply it as defaults before the real parse
    pre_args, _ = argument_parser.parse_known_args()
    if pre_args.config is not None:
        with open(pre_args.config) as f:
            config = json.load(f)
        argument_parser.set_defaults(**config)

    args = argument_parser.parse_args()

    required = ["input_image", "triangles", "population_size", "generations", "k"]
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        argument_parser.error(f"the following arguments are required (or set them in --config): {', '.join(missing)}")

    return args
