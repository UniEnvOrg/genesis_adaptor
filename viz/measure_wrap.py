"""How far around each link the sensed band actually reaches."""
# arc_degrees spans probe *centres*; each probe also senses its own spherical
# footprint, so the real band is wider. Samples the surface, keeps points
# within probe_radius of a probe, and reports the arc about the long axis.

import numpy as np
import trimesh

from common import (
    ARC, N_ALONG, N_AROUND, SENSED as NAMES, arc_probes, build_hand,
    face_axis_for, link_frame,
    link_mesh,
)

PROBE_RADIUS = 0.004      # keep in step with TactileSensorNodeOptions.probe_radius
N_SAMPLES = 120_000
SLAB = 0.004              # half-thickness of the cross-section slab sampled, m

env, e = build_hand()


def sensed_band(mesh, probes, radius, face_axis):
    """Angular span covered by the union of probe footprints, about the long axis."""
    f = link_frame(mesh, face_axis)
    la, u, v, c = f["la"], f["u"], f["v"], f["centre"]

    # sample the cross-section through a probe row, not through the link's
    # midpoint -- rows are inset by `margin`, so the midpoint can fall between
    # two rows and register nothing (the palm is long enough for that to happen)
    rows = np.unique(np.round(probes[:, la], 6))
    pts, _ = trimesh.sample.sample_surface(mesh, N_SAMPLES)
    band = pts[np.abs(pts[:, la] - rows[len(rows) // 2]) < SLAB]
    if len(band) < 50:
        return float("nan")
    ang = np.degrees(np.arctan2(band[:, v] - c[v], band[:, u] - c[u]))
    hit = np.linalg.norm(band[:, None, :] - probes[None, :, :], axis=2).min(axis=1) <= radius
    if not hit.any():
        return 0.0
    a = np.sort(ang[hit])
    gaps = np.diff(np.concatenate([a, [a[0] + 360]]))
    return float(360 - gaps.max())


print(f"{N_AROUND}x{N_ALONG} taxels, arc_degrees={ARC:.0f}, probe_radius={PROBE_RADIUS*1000:.0f} mm")
print(f"{'link':24s} {'sensed band':>12s}")
print("-" * 38)
bands = []
for n in NAMES:
    lk = e.get_link(n)
    pos, _ = arc_probes(lk)
    b = sensed_band(link_mesh(lk), pos, PROBE_RADIUS, face_axis_for(n))
    bands.append(b)
    print(f"{n:24s} {b:11.0f}°")

arr = np.array(bands[1:])
print("-" * 38)
print(f"finger segments {arr.min():.0f}-{arr.max():.0f}° (mean {arr.mean():.0f}°), palm {bands[0]:.0f}°")
env.close()
