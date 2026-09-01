"""Build the hand once, then emit everything the page needs from it.

Merges what used to be ``project_faces`` and ``render_hand``: both wanted the
same scene and the same link meshes, and building the scene is expensive, so it
now happens once. Writes ``out/taxels.json`` and ``out/hand_*.jpg``, and returns
both so ``build.py`` can inline them without a second pass over the disk.

The renders rasterise the hand's collision geometry -- the surface the contact
probes actually act on -- and shade every pixel by which taxel owns it.
Ownership is the probe's Voronoi cell in the link's own (angle-about-long-axis,
along-axis) coordinates, exactly the coordinates ``_arc_probes`` places probes
in, so the tiles curve with the link instead of being drawn flat.
"""
import io
import json
from typing import NamedTuple

import numpy as np
from PIL import Image

from common import (
    ARC, MARGIN, N_ALONG, N_AROUND, PLAIN, SENSED, SIDE,
    build_hand, face_axis_for, link_frame, link_mesh, out_path, to_link, to_world,
    view_matrix, world_T,
)

SS = 2                                   # supersample factor
W, H = 1150, 1300
FOV = 30
JPEG_QUALITY = 88

C_SKIN = np.array([0.63, 0.67, 0.70])
C_TAX  = np.array([0.93, 0.60, 0.20])
C_SEAM = np.array([0.31, 0.17, 0.03])


class Hand(NamedTuple):
    """The whole hand as one triangle soup, in world and link coordinates.

    ``FL`` labels each triangle with its index into ``params``, or -1 for a link
    that carries no taxels; ``params[k]`` is None for those.
    """
    V_w: np.ndarray
    V_l: np.ndarray
    F: np.ndarray
    FL: np.ndarray
    params: list


# ---------------------------------------------------------------- geometry ---
def band_params(mesh, face_axis):
    """`link_frame` plus where the probe rows and columns land in it."""
    f = link_frame(mesh, face_axis)
    span = f["hi"][f["la"]] - f["lo"][f["la"]]
    stations = np.linspace(
        f["lo"][f["la"]] + MARGIN * span, f["hi"][f["la"]] - MARGIN * span, N_ALONG
    )
    half = np.radians(ARC) / 2.0
    angles = np.linspace(-half, half, N_AROUND)
    return dict(f, stations=stations, angles=angles,
                d_s=float(stations[1] - stations[0]),
                d_a=float(angles[1] - angles[0]))


def face_row(name, lk, mesh, palm):
    """The link's sensed face projected into the palm plane: size, angle, centre, area."""
    lo, hi = mesh.bounds
    area = float((hi[1] - lo[1]) * (hi[2] - lo[2]) * 1e4)          # cm^2

    # the face centre, and a point 10 mm along the link, both in palm coordinates:
    # together they give where the face sits and which way it points once projected
    ctr_l = np.array([hi[0], 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])])
    ctr_p, zt_p = to_link(to_world([ctr_l, ctr_l + [0.0, 0.0, 0.01]], lk), palm)
    ang = float(np.degrees(np.arctan2(zt_p[2] - ctr_p[2], zt_p[1] - ctr_p[1])))

    return {"link": name, "area_cm2": area,
            "w": float((hi[1] - lo[1]) * 1000), "h": float((hi[2] - lo[2]) * 1000),
            "ang": ang, "c": [float(ctr_p[1] * 1000), float(ctr_p[2] * 1000)]}


def collect(ent, palm):
    """One pass over the links: the render's triangle soup and the face table."""
    verts_w, verts_l, faces, fl = [], [], [], []
    params, rows, base = [], [], 0
    for k, n in enumerate(SENSED + PLAIN):
        lk = ent.get_link(n)
        if not lk.geoms:
            continue
        m = link_mesh(lk)
        vl = np.asarray(m.vertices, float)
        verts_l.append(vl)
        verts_w.append(to_world(vl, lk))
        faces.append(np.asarray(m.faces) + base)
        fl.append(np.full(len(m.faces), k if n in SENSED else -1))
        params.append(band_params(m, face_axis_for(n)) if n in SENSED else None)
        base += len(vl)

        if n in SENSED:
            rows.append(face_row(n, lk, m, palm))
            print(f"{n:26s} area {rows[-1]['area_cm2']:5.1f} cm^2")

    hand = Hand(np.concatenate(verts_w), np.concatenate(verts_l),
                np.concatenate(faces), np.concatenate(fl), params)
    print(f"{len(hand.F)} triangles, {len(hand.V_w)} vertices")
    return hand, rows


# ------------------------------------------------------------- rasteriser ---
def fit_eye(V_w, direction, target, fov_deg, w, h, up, pad=1.06):
    """Push the eye back along `direction` until every vertex is inside the frame."""
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    s_ = np.cross(-d, up); s_ /= np.linalg.norm(s_)
    u_ = np.cross(s_, -d)
    tv = np.tan(np.radians(fov_deg) / 2) / pad
    th = tv * w / h
    rel = V_w - target
    a, x, y = rel @ d, rel @ s_, rel @ u_
    return target + d * float(np.max(np.maximum(a + np.abs(x) / th, a + np.abs(y) / tv)))


def render(hand, direction, target, fov_deg, w, h, up=None):
    V_w, V_l, F, FL, params = hand
    up = np.array([0.0, 0.0, 1.0]) if up is None else up
    eye = fit_eye(V_w, direction, np.asarray(target, float), fov_deg, w, h, up)
    Vm = view_matrix(eye, target, up)
    cam = (Vm[:3, :3] @ V_w.T).T + Vm[:3, 3]           # camera space, -z forward
    depth = -cam[:, 2]
    fpx = (h / 2) / np.tan(np.radians(fov_deg) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = w / 2 + cam[:, 0] / depth * fpx
        sy = h / 2 - cam[:, 1] / depth * fpx
    inv = 1.0 / np.maximum(depth, 1e-6)

    zbuf = np.full((h, w), np.inf, np.float32)
    lid = np.full((h, w), -2, np.int16)
    lpos = np.zeros((h, w, 3), np.float32)
    nrm = np.zeros((h, w, 3), np.float32)

    tri_w = V_w[F]
    fn = np.cross(tri_w[:, 1] - tri_w[:, 0], tri_w[:, 2] - tri_w[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)

    ok = np.all(depth[F] > 1e-4, axis=1)
    for t in np.flatnonzero(ok):
        i0, i1, i2 = F[t]
        x0, x1, x2 = sx[i0], sx[i1], sx[i2]
        y0, y1, y2 = sy[i0], sy[i1], sy[i2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        xlo = max(int(np.floor(min(x0, x1, x2))), 0)
        xhi = min(int(np.ceil(max(x0, x1, x2))), w - 1)
        ylo = max(int(np.floor(min(y0, y1, y2))), 0)
        yhi = min(int(np.ceil(max(y0, y1, y2))), h - 1)
        if xhi < xlo or yhi < ylo:
            continue
        xs = np.arange(xlo, xhi + 1) + 0.5
        ys = np.arange(ylo, yhi + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        l0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area
        l1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        wsum = l0 * inv[i0] + l1 * inv[i1] + l2 * inv[i2]
        z = 1.0 / np.maximum(wsum, 1e-12)
        sub = zbuf[ylo:yhi + 1, xlo:xhi + 1]
        hit = inside & (z < sub)
        if not hit.any():
            continue
        a0 = (l0 * inv[i0] / wsum)[hit]
        a1 = (l1 * inv[i1] / wsum)[hit]
        a2 = (l2 * inv[i2] / wsum)[hit]
        p = (V_l[i0] * a0[:, None] + V_l[i1] * a1[:, None] + V_l[i2] * a2[:, None])
        sub[hit] = z[hit]
        lid[ylo:yhi + 1, xlo:xhi + 1][hit] = FL[t]
        lpos[ylo:yhi + 1, xlo:xhi + 1][hit] = p
        nrm[ylo:yhi + 1, xlo:xhi + 1][hit] = fn[t]

    # ------------------------------------------------------------- shading ---
    alb = np.zeros((h, w, 3), np.float32)
    body = lid > -2
    alb[body] = C_SKIN
    for k, pr in enumerate(params):
        if pr is None:
            continue
        sel = lid == k
        if not sel.any():
            continue
        p = lpos[sel]
        cu = (p[:, pr["u"]] - pr["centre"][pr["u"]]) * pr["sign"]
        cv = p[:, pr["v"]] - pr["centre"][pr["v"]]
        th = np.arctan2(cv, cu)
        s = p[:, pr["la"]]
        ja = np.abs(th[:, None] - pr["angles"][None, :]).argmin(1)
        ia = np.abs(s[:, None] - pr["stations"][None, :]).argmin(1)
        fa = (th - pr["angles"][ja]) / pr["d_a"]
        fs = (s - pr["stations"][ia]) / pr["d_s"]
        on = (np.abs(fa) <= 0.5) & (np.abs(fs) <= 0.5)
        seam = np.maximum(np.abs(fa), np.abs(fs)) > 0.5 - 0.032
        col = np.repeat(C_SKIN[None, :], len(p), 0).astype(np.float32)
        col[on] = C_TAX
        col[on & seam] = C_SEAM
        alb[sel] = col

    # lights ride with the camera, so every view is lit the same way
    back = np.asarray(direction, float); back /= np.linalg.norm(back)   # target -> eye
    right = np.cross(-back, up); right /= np.linalg.norm(right)
    upc = np.cross(right, -back)
    key = back * 0.80 + upc * 0.42 + right * 0.32; key /= np.linalg.norm(key)
    fill = back * 0.45 - right * 0.72 - upc * 0.20; fill /= np.linalg.norm(fill)
    nl = np.clip(nrm @ key, 0, 1) * 0.62 + np.clip(nrm @ fill, 0, 1) * 0.22 + 0.36
    img = np.clip(alb * nl[..., None], 0, 1)
    img[~body] = 0.965                                   # paper-white ground
    return (img ** (1 / 1.05) * 255).astype(np.uint8)


def encode(img) -> bytes:
    """Downsample the supersampled render and encode it once, as JPEG."""
    buf = io.BytesIO()
    im = Image.fromarray(img).resize((W, H), Image.LANCZOS).convert("RGB")
    im.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


# ------------------------------------------------------------------- bake ---
def bake():
    """Build the hand, measure its faces, render the views.

    Returns ``(rows, images)`` -- the face table and ``{tag: JPEG bytes}`` -- and
    writes both to ``out/`` on the way through.
    """
    env, ent = build_hand()
    palm = ent.get_link(f"{SIDE}_wuji_palm")
    hand, rows = collect(ent, palm)
    Tp = world_T(palm)
    px, py, pz = Tp[:3, 0], Tp[:3, 1], Tp[:3, 2]
    env.close()

    json.dump(rows, open(out_path("taxels.json"), "w"), indent=1)
    print("wrote taxels.json")

    ctr = 0.5 * (hand.V_w.min(0) + hand.V_w.max(0))
    views = {
        "hand_palm":          px,
        "hand_three_quarter": px * .62 + py * .78,
        "hand_edge":          px * .12 + py * .99 + pz * .10,
    }
    images = {}
    for tag, dirn in views.items():
        images[tag] = encode(render(hand, dirn, ctr, FOV, W * SS, H * SS, up=pz))
        open(out_path(f"{tag}.jpg"), "wb").write(images[tag])
        print(f"wrote {tag}.jpg  ({len(images[tag]) // 1024} KB)")
    return rows, images


if __name__ == "__main__":
    bake()
