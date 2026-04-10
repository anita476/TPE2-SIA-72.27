import sys
from io import BytesIO

from PIL import Image

from utils.dispatch import CROSSOVER_MAP, FITNESS_MAP, MUTATION_MAP, SURVIVAL_MAP, build_selector, build_stop_condition
from genetic_algorithm import run_genetic_algorithm
from input_output_handler import read_image, save_image, parse_arguments


def main():
    args = parse_arguments()

    try:
        image_bytes = read_image(args.input_image)
    except (FileNotFoundError, OSError) as e:
        print(f"Error reading image: {e}")
        sys.exit(1)

    with Image.open(BytesIO(image_bytes)) as img:
        source_image = img.convert("RGBA").copy()

    result = run_genetic_algorithm(
        source_image=source_image,
        num_triangles=args.triangles,
        population_size=args.population_size,
        generations=args.generations,
        k=args.k,
        selector=build_selector(args.selector, args.temperature, args.temperature_min, args.temperature_decay),
        crossover=CROSSOVER_MAP[args.crossover],
        fitness_fn=FITNESS_MAP[args.fitness],
        survival_strategy=SURVIVAL_MAP[args.survival_strategy],
        mutation_fn=MUTATION_MAP[args.mutation],
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        snapshot_interval=args.snapshot_interval,
        output_dir=args.output_dir,
        seed=args.seed,
        stop_condition=build_stop_condition(
            args.target_fitness,
            args.convergence_window,
            args.convergence_delta,
            args.time_limit,
        ),
    )

    try:
        save_image(result.image_bytes, args.output_image)
    except OSError as e:
        print(f"Error saving image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
