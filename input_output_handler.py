import argparse
import os


def read_image(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            image = f.read()
            print(f'Image read: "{os.path.basename(path)}", {len(image)} bytes')
            return image
    except Exception as e:
        print(f"Error reading image: {e}")
        exit(1)


def save_image(image, path: str) -> bool:
    try:
        with open(path, "wb") as f:
            f.write(image)
            print(f'Image saved: "{os.path.basename(path)}", {len(image)} bytes')
            return True
    except Exception as e:
        print(f"Error saving image: {e}")
        exit(1)


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(description="Genetics Algorithm Engine that receives an image and processes it into a triangle aproximation.")
    argument_parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    argument_parser.add_argument("--output_image", type=str, default="output.png", help="Path to the output image")
    argument_parser.add_argument("--triangles", type=int, required=True, help="Triangles to use in the approximation")
    return argument_parser.parse_args()
