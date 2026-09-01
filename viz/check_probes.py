"""Validate _arc_probes on the real Wuji links: on-surface, unit outward normals, band width."""
import numpy as np
import trimesh

from common import SENSED as NAMES, arc_probes, build_hand, link_frame, link_mesh

env, e = build_hand()
print(f"{'link':24s} {'N':>3s} {'off-surface mm':>15s} {'|n|':>6s} {'outward':>8s} {'band°':>7s}")
print("-" * 72)
worst_off = 0.0
for n in NAMES:
    lk = e.get_link(n)
    pos, nrm = arc_probes(lk)
    mesh = link_mesh(lk)
    d = trimesh.proximity.closest_point(mesh, pos)[1]
    worst_off = max(worst_off, float(d.max()))

    f = link_frame(mesh)
    la, u, v, c = f["la"], f["u"], f["v"], f["centre"]
    rad = pos - c
    rad[:, la] = 0.0
    outward = float((np.einsum("ij,ij->i", nrm, rad) > 0).mean())
    ang = np.degrees(np.arctan2(pos[:, v] - c[v], pos[:, u] - c[u]))
    band = float(ang.max() - ang.min())
    print(f"{n:24s} {len(pos):3d} {d.max()*1000:15.3f} "
          f"{np.linalg.norm(nrm,axis=1).mean():6.3f} {outward*100:7.0f}% {band:7.0f}")

print("-" * 72)
print(f"worst off-surface distance across all links: {worst_off*1000:.3f} mm")
env.close()
