"""Wuji hand taxel layout, unrolled into the palm plane."""
# Each sensing face is drawn at true size, projected into the palm plane, then
# tiled n_around x n_along -- one tile per taxel, edge to edge. Every finger's
# link1 has no collision geometry and so no taxels; a connector bridges it.
import json, math
import numpy as np

from common import N_ALONG, N_AROUND, out_path

N = N_AROUND * N_ALONG

FILL = "#e08a2c"
SEAM = "#b26a1c"     # hairline between taxels of the same link
EDGE = "#8a4e12"     # link outline
PAD = 2.2            # mm of overlap so segment seams have no gap
REACH = 52           # mm the connector runs back toward the palm


def frame(r, pad=PAD):
    """Face centre, along-axis unit vector, across unit vector, and padded size."""
    a = math.radians(r["ang"])
    u = np.array([math.cos(a), math.sin(a)])     # along the link's long axis
    v = np.array([-u[1], u[0]])                  # across it (the wrapped arc)
    return np.array(r["c"], float), u, v, r["h"] + pad * 2, r["w"] + pad * 2


def rect(r):
    c, u, v, H, W = frame(r)
    return [c + u * H * su + v * W * sv
            for su, sv in ((-.5, -.5), (-.5, .5), (.5, .5), (.5, -.5))]


def tiles(r):
    """The face cut into N_ALONG rows along the link by N_AROUND columns across it."""
    c, u, v, H, W = frame(r)
    out = []
    for i in range(N_ALONG):
        for j in range(N_AROUND):
            su0, su1 = i / N_ALONG - .5, (i + 1) / N_ALONG - .5
            sv0, sv1 = j / N_AROUND - .5, (j + 1) / N_AROUND - .5
            quad = [c + u * H * su + v * W * sv
                    for su, sv in ((su0, sv0), (su0, sv1), (su1, sv1), (su1, sv0))]
            out.append((quad, i * N_AROUND + j))
    return out


def connector(quad, palm_c):
    p = np.array(quad, dtype=float)
    order = sorted(range(4), key=lambda i: np.linalg.norm(p[i] - palm_c))
    a, b = p[order[0]], p[order[1]]
    mid = (a + b) / 2
    dv = palm_c - mid
    dv /= max(np.linalg.norm(dv), 1e-9)
    return [a, b, b + dv * REACH, a + dv * REACH]


def draw_layout(rows) -> str:
    """The face table from ``bake`` -> the palmar schematic, as SVG source."""
    palm_row = next(r for r in rows if r["link"].endswith("wuji_palm"))
    palm_c = np.array(palm_row["c"], dtype=float)

    quads = [(rect(r), r) for r in rows]
    conns = [connector(q, palm_c) for q, r in quads if r["link"].endswith("link2")]

    allp = np.concatenate([np.array(q) for q, _ in quads] + [np.array(c) for c in conns])
    lo, hi = allp.min(axis=0) - 6, allp.max(axis=0) + 6
    w, h = hi - lo

    def xy(p):
        return (hi[0] - p[0], hi[1] - p[1])      # mirror for handedness, flip for up

    def pts(q):
        return " ".join(f"{xy(v)[0]:.1f},{xy(v)[1]:.1f}" for v in q)

    parts = [f'<polygon points="{pts(c)}" fill="{FILL}"></polygon>' for c in conns]
    for _, r in quads:
        per = r["area_cm2"] / N
        for quad, k in tiles(r):
            parts.append(
                f'<polygon class="tx" points="{pts(quad)}" fill="{FILL}" stroke="{SEAM}" '
                f'stroke-width="0.45" data-l="{r["link"]} [{k}]" data-a="{per:.2f}"></polygon>'
            )
    # link outlines last, so each link still reads as one part
    parts += [f'<polygon points="{pts(q)}" fill="none" stroke="{EDGE}" stroke-width="1" '
              f'stroke-linejoin="round"></polygon>' for q, _ in quads]

    svg = (
        f'<svg viewBox="0 0 {w:.1f} {h:.1f}" role="img" '
        f'aria-label="Wuji hand taxel layout, palmar view: the palm plate plus three '
        f'segments on each of the five fingers, each tiled into a {N_AROUND} by {N_ALONG} '
        f'grid of {N} taxels filled across their full sensing area">'
        + "".join(parts) + "</svg>"
    )
    print(f"{w:.0f} x {h:.0f} mm | {len(quads)} links x {N} taxels = {len(quads)*N} | "
          f"{len(conns)} connectors | {len(svg)//1024} KB")
    return svg


if __name__ == "__main__":
    svg = draw_layout(json.load(open(out_path("taxels.json"))))
    open(out_path("hand_layout.svg"), "w").write(svg)
