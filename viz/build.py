import base64
from pathlib import Path

from artifacts import build_static_artifacts
from common import N_ALONG, N_AROUND, PROBE_RADIUS, SENSED, get_output_path
from draw_layout import draw_layout

TEMPLATE_PATH = Path(__file__).with_name("template.html")
TAXELS_PER_LINK = N_AROUND * N_ALONG
LINKS_PER_HAND = len(SENSED)
TAXELS_PER_HAND = TAXELS_PER_LINK * LINKS_PER_HAND


def _format_range(values, precision):
    lower = f"{min(values):.{precision}f}"
    upper = f"{max(values):.{precision}f}"
    return lower if lower == upper else f"{lower} &ndash; {upper}"


def _render_area_table(rows):
    groups = (
        ("palm plate", "wuji_palm"),
        ("finger <code>link2</code>", "link2"),
        ("finger <code>link3</code>", "link3"),
        ("finger <code>link4</code>", "link4"),
    )
    table_rows = []
    for label, suffix in groups:
        areas = [row["area_cm2"] for row in rows if row["link"].endswith(suffix)]
        if areas:
            per_taxel = [area / TAXELS_PER_LINK for area in areas]
            table_rows.append(
                f"<tr><td>{label}</td><td>{len(areas)}</td><td>{_format_range(areas, 1)}</td>"
                f"<td>{_format_range(per_taxel, 2)}</td></tr>"
            )
    return "\n".join(table_rows)


def _render_page(rows, images, svg):
    values = {
        "HAND": svg,
        "N_AROUND": N_AROUND,
        "N_ALONG": N_ALONG,
        "N_PER_LINK": TAXELS_PER_LINK,
        "N_LINKS": LINKS_PER_HAND,
        "N_PER_HAND": TAXELS_PER_HAND,
        "N_BOTH": TAXELS_PER_HAND * 2,
        "PROBE_RADIUS_MM": f"{PROBE_RADIUS * 1000:g}",
        "AREA_TABLE": _render_area_table(rows),
    }
    page = TEMPLATE_PATH.read_text()
    for key, value in values.items():
        page = page.replace(f"__{key}__", str(value))
    for name, image in images.items():
        page = page.replace(f"__{name.upper()}__", base64.b64encode(image).decode())
    return page


def main():
    rows, images = build_static_artifacts()
    svg = draw_layout(rows)
    Path(get_output_path("hand_layout.svg")).write_text(svg)
    page = _render_page(rows, images, svg)
    path = Path(get_output_path("tactile.html"))
    path.write_text(page)
    print(f"wrote {path}  ({len(page) // 1024} KB, {N_AROUND}x{N_ALONG} taxels per link)")


if __name__ == "__main__":
    main()
