import os
from input_output_handler import read_image, save_image, parse_arguments
from image_processor import process_image


def main():
    args = parse_arguments()
    image_path = args.input_image
    image = read_image(image_path)
    processed_image = process_image(image, args.triangles)
    save_image(processed_image, args.output_image)


if __name__ == "__main__":
    main()
