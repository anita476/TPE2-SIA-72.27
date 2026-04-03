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
    argument_parser = argparse.ArgumentParser(description="Genetics Algorithm Engine that receives an image and processes it into a triangle aproximation.")
    argument_parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    argument_parser.add_argument("--output_image", type=str, default="output.png", help="Path to the output image")
    argument_parser.add_argument("--triangles", type=int, required=True, help="Triangles to use in the approximation")
    argument_parser.add_argument("--population-size", type=int, required=True, help="Number of individuals in the population")
    argument_parser.add_argument("--generations", type=int, required=True, help="Number of generations to run")
    argument_parser.add_argument("--individuals-kept", type=float, default=0.2, help="Fraction of top individuals kept (0.0 to 1.0)")
    argument_parser.add_argument("--mutation-rate", type=float, default=0.1, help="Probability of mutating each triangle (0.0 to 1.0)")
    argument_parser.add_argument("--mutation-strength", type=float, default=0.3, help="Magnitude of each mutation (0.0 to 1.0)")
    argument_parser.add_argument("--snapshot-interval", type=int, default=0, help="Save snapshot every N generations (0=disabled)")
    argument_parser.add_argument("--output-dir", type=str, default="snapshots", help="Directory for snapshot images")
    return argument_parser.parse_args()
