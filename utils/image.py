from __future__ import annotations

import os
import numpy as np
from PIL import Image, ImageDraw

from .genetic import Color, Genotype


def create_phenotype_image(
    genotypes: list[Genotype],
    image_size: tuple[int, int],
    background: Color = (255, 255, 255, 255),
) -> Image.Image:
    canvas = Image.new("RGBA", image_size, background)
    draw = ImageDraw.Draw(canvas, "RGBA")

    for genotype in genotypes:
        draw.polygon(genotype.vertices, fill=genotype.color)

    return canvas


def save_phenotype_image(best_individual: list[Genotype], output_dir: str, gen: int, width: int, height: int) -> None:
    """Render the individual as an image and save it as a snapshot PNG."""
    snapshot_img = create_phenotype_image(best_individual, image_size=(width, height))
    snapshot_path = os.path.join(output_dir, f"gen_{gen + 1:06d}.png")
    snapshot_img.save(snapshot_path, format="PNG")
    print(f'Snapshot saved: "{os.path.basename(snapshot_path)}"')


def compute_mae(source_array: np.ndarray, candidate: Image.Image) -> float:
    """Compute Mean Absolute Error (MAE) between a pre-converted source array and a candidate image."""
    arr2 = np.asarray(candidate.convert("RGBA"), dtype=np.int16)
    return float(np.mean(np.abs(source_array - arr2)))
