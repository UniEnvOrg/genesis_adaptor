import json
from pathlib import Path

from common import N_ALONG, N_AROUND, get_output_path
from layout import compute_bounds, compute_connector, compute_rectangle, generate_tiles, transform_to_canvas

TAXELS_PER_LINK = N_AROUND * N_ALONG
TAXEL_COLOR = "#e08a2c"
SEAM_COLOR = "#b26a1c"
OUTLINE_COLOR = "#8a4e12"


def _format_points(points, upper_bound):
    canvas_points = transform_to_canvas(points, upper_bound)
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in canvas_points)


def draw_layout(rows) -> str:
    palm = next(row for row in rows if row["link"].endswith("wuji_palm"))
    palm_centre = palm["c"]
    rectangles = [(compute_rectangle(row), row) for row in rows]
    connectors = [
        compute_connector(rectangle, palm_centre)
        for rectangle, row in rectangles
        if row["link"].endswith("link2")
    ]
    lower, upper = compute_bounds([rectangle for rectangle, _ in rectangles] + connectors)
    width, height = upper - lower

    parts = [
        f'<polygon points="{_format_points(connector, upper)}" fill="{TAXEL_COLOR}"></polygon>'
        for connector in connectors
    ]
    for _, row in rectangles:
        area = row["area_cm2"] / TAXELS_PER_LINK
        for index, tile in generate_tiles(row, N_AROUND, N_ALONG):
            points = _format_points(tile, upper)
            parts.append(
                f'<polygon class="tx" points="{points}" fill="{TAXEL_COLOR}" stroke="{SEAM_COLOR}" '
                f'stroke-width="0.45" data-l="{row["link"]} [{index}]" data-a="{area:.2f}"></polygon>'
            )
    parts.extend(
        f'<polygon points="{_format_points(rectangle, upper)}" fill="none" stroke="{OUTLINE_COLOR}" '
        f'stroke-width="1" stroke-linejoin="round"></polygon>'
        for rectangle, _ in rectangles
    )

    aria_label = (
        "Wuji hand taxel layout, palmar view: the palm plate plus three segments on each of the five "
        f"fingers, each tiled into a {N_AROUND} by {N_ALONG} grid of {TAXELS_PER_LINK} taxels filled "
        "across their full sensing area"
    )
    svg = (
        f'<svg viewBox="0 0 {width:.1f} {height:.1f}" role="img" aria-label="{aria_label}">'
        + "".join(parts)
        + "</svg>"
    )
    print(
        f"{width:.0f} x {height:.0f} mm | {len(rectangles)} links x {TAXELS_PER_LINK} taxels = "
        f"{len(rectangles) * TAXELS_PER_LINK} | {len(connectors)} connectors | {len(svg) // 1024} KB"
    )
    return svg


if __name__ == "__main__":
    rows = json.loads(Path(get_output_path("taxels.json")).read_text())
    Path(get_output_path("hand_layout.svg")).write_text(draw_layout(rows))
