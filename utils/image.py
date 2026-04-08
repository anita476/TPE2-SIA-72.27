from __future__ import annotations

import os
import numpy as np
from PIL import Image, ImageDraw

from .genetic import Color, Gene


def render_individual_visible_array(
    genes: list[Gene],
    image_size: tuple[int, int],
    background: Color = (255, 255, 255, 255),
) -> np.ndarray:
    canvas = create_phenotype_image(genes, image_size=image_size, background=background)
    array = np.asarray(canvas, dtype=np.uint8)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def visible_array_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array, "RGB")


def create_phenotype_image(
    genes: list[Gene],
    image_size: tuple[int, int],
    background: Color = (255, 255, 255, 255),
) -> Image.Image:
    canvas = Image.new("RGB", image_size, background[:3])
    draw = ImageDraw.Draw(canvas, "RGBA")

    for gene in genes:
        if gene.color[3] <= 0:
            continue
        draw.polygon(gene.vertices, fill=gene.color)

    return canvas


def save_phenotype_image(best_individual: list[Gene], output_dir: str, gen: int, width: int, height: int) -> None:
    """Render the individual as an image and save it as a snapshot PNG."""
    snapshot_img = create_phenotype_image(best_individual, image_size=(width, height))
    snapshot_path = os.path.join(output_dir, f"gen_{gen + 1:06d}.png")
    snapshot_img.save(snapshot_path, format="PNG")
    print(f'Snapshot saved: "{os.path.basename(snapshot_path)}"')
