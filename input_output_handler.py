import argparse
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
    argument_parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    argument_parser.add_argument("--output_image", type=str, default="output.png", help="Path to the output image")
    argument_parser.add_argument("--triangles", type=int, required=True, help="Triangles to use in the approximation")
    argument_parser.add_argument("--population-size", type=int, required=True, help="Number of individuals in the population")
    argument_parser.add_argument("--generations", type=int, required=True, help="Number of generations to run")
    argument_parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of selected parents and offspring generated per generation",
    )
    argument_parser.add_argument("--mutation-rate", type=float, default=0.1, help="Probability of mutating each triangle (0.0 to 1.0)")
    argument_parser.add_argument("--mutation-strength", type=float, default=0.3, help="Magnitude of each mutation (0.0 to 1.0)")
    argument_parser.add_argument("--snapshot-interval", type=int, default=0, help="Save snapshot every N generations (0=disabled)")
    argument_parser.add_argument("--output-dir", type=str, default="snapshots", help="Directory for snapshot images")
    argument_parser.add_argument("--selector", type=str, default="elite", choices=["elite", "roulette", "universal", "ranking", "boltzmann", "tournament_det", "tournament_stoch"], help="Selection method (elite, roulette, universal, ranking, boltzmann, tournament deterministic and stochastic)")
    argument_parser.add_argument("--crossover", type=str, default="two_point", choices=["one_point", "two_point"], help="Crossover operator (one_point, two_point)")
    argument_parser.add_argument(
        "--fitness",
        type=str,
        default="mae",
        choices=["mae", "mse", "rmse"],
        help="Fitness key (maps to mae_fitness, mse_fitness, rmse_fitness); RGBA, lower is better",
    )
    argument_parser.add_argument(
        "--mutation",
        type=str,
        default="multigen_uniform",
        choices=["gen", "multigen_limited", "multigen_uniform", "complete"],
        help="Mutation strategy: gen (single gene), multigen_limited (random subset), multigen_uniform (each gene independently), complete (all genes)",
    )
    argument_parser.add_argument(
        "--survival_strategy",
        type=str,
        default="additive",
        choices=["additive", "exclusive"],
        help="Survival strategy for the next generation: additive or exclusive",
    )
    argument_parser.add_argument("--temperature", type=float, default=50.0, help="Initial temperature T_0 for Boltzmann selection")
    argument_parser.add_argument("--temperature-min", type=float, default=1.0, help="Convergence temperature T_c for Boltzmann selection")
    argument_parser.add_argument("--temperature-decay", type=float, default=-0.005, help="Decay constant k for Boltzmann schedule: T(t) = T_c + (T_0 - T_c) * e^(k*t), k < 0 for cooling")
    return argument_parser.parse_args()
