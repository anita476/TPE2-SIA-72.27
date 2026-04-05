from __future__ import annotations

import os
from PIL import Image, ImageDraw

from .genetic import Color, Gene


def create_phenotype_image(
    genes: list[Gene],
    image_size: tuple[int, int],
    background: Color = (255, 255, 255, 255),
) -> Image.Image:
    canvas = Image.new("RGBA", image_size, background)
    draw = ImageDraw.Draw(canvas, "RGBA")

    for gene in genes:
        draw.polygon(gene.vertices, fill=gene.color)

    return canvas


def save_phenotype_image(best_individual: list[Gene], output_dir: str, gen: int, width: int, height: int) -> None:
    """Render the individual as an image and save it as a snapshot PNG."""
    snapshot_img = create_phenotype_image(best_individual, image_size=(width, height))
    snapshot_path = os.path.join(output_dir, f"gen_{gen + 1:06d}.png")
    snapshot_img.save(snapshot_path, format="PNG")
    print(f'Snapshot saved: "{os.path.basename(snapshot_path)}"')
