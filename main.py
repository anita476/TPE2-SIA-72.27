import sys
from io import BytesIO
from PIL import Image
from input_output_handler import read_image, save_image, parse_arguments
from crossovers import one_point_crossover, two_point_crossover
from genetic_algorithm import run_genetic_algorithm
from selectors import elite_selection, roulette_selection, universal_selection

_SELECTORS = {
    "elite": elite_selection,
    "roulette": roulette_selection,
    "universal": universal_selection,
}

_CROSSOVERS = {
    "one_point": one_point_crossover,
    "two_point": two_point_crossover,
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

    result = run_genetic_algorithm(
        source_image=source_image,
        num_triangles=args.triangles,
        population_size=args.population_size,
        generations=args.generations,
        k=args.k,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        snapshot_interval=args.snapshot_interval,
        output_dir=args.output_dir,
        selector=_SELECTORS[args.selector],
        crossover=_CROSSOVERS[args.crossover],
    )

    try:
        save_image(result, args.output_image)
    except OSError as e:
        print(f"Error saving image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
