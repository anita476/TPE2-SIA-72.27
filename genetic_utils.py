from __future__ import annotations

from dataclasses import dataclass
import random


Color = tuple[int, int, int, int]
Vertex = tuple[int, int]
Vertices = tuple[Vertex, Vertex, Vertex]

OVERFLOW_MARGIN_RATIO = 0.2


@dataclass(slots=True, frozen=True)
class Genotype:
    color: Color
    vertices: Vertices


def get_overflow_bounds(width: int, height: int, delta: float = OVERFLOW_MARGIN_RATIO) -> tuple[int, int, int, int]:
    margin_x = int(width * delta)
    margin_y = int(height * delta)

    min_x = -margin_x
    max_x = width + margin_x

    min_y = -margin_y
    max_y = height + margin_y

    return min_x, max_x, min_y, max_y


def build_genotype(
    color: Color,
    vertices: Vertices,
) -> Genotype:
    if len(color) != 4:
        raise ValueError("Color must contain exactly 4 values: r, g, b, a")

    if len(vertices) != 3:
        raise ValueError("Vertices must contain exactly 3 positions")

    return Genotype(color=color, vertices=vertices)


def mutate_genotype(
    genotype: Genotype,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    strength: float,
) -> Genotype:
    def clamp(value: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, value))

    r, g, b, a = genotype.color
    delta = int(255 * strength)
    color = (
        clamp(r + rng.randint(-delta, delta), 0, 255),
        clamp(g + rng.randint(-delta, delta), 0, 255),
        clamp(b + rng.randint(-delta, delta), 0, 255),
        clamp(a + rng.randint(-delta, delta), 0, 255),
    )

    min_x, max_x_bound, min_y, max_y_bound = bounds
    dx = int((max_x_bound - min_x) * strength)
    dy = int((max_y_bound - min_y) * strength)
    vertices = tuple(
        (
            clamp(vx + rng.randint(-dx, dx), min_x, max_x_bound - 1),
            clamp(vy + rng.randint(-dy, dy), min_y, max_y_bound - 1),
        )
        for vx, vy in genotype.vertices
    )

    return Genotype(color=color, vertices=vertices)


def random_triangle(
    max_x: int,
    max_y: int,
    rng: random.Random | None = None,
) -> Genotype:
    generator = rng if rng is not None else random.Random()

    color = (generator.randint(0, 255), generator.randint(0, 255), generator.randint(0, 255), generator.randint(0, 255))

    min_x_margin, max_x_margin, min_y_margin, max_y_margin = get_overflow_bounds(max_x, max_y)
    vertices = (
        (generator.randrange(min_x_margin, max_x_margin), generator.randrange(min_y_margin, max_y_margin)),
        (generator.randrange(min_x_margin, max_x_margin), generator.randrange(min_y_margin, max_y_margin)),
        (generator.randrange(min_x_margin, max_x_margin), generator.randrange(min_y_margin, max_y_margin)),
    )

    return build_genotype(color=color, vertices=vertices)
