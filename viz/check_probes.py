import numpy as np
import trimesh

from common import SENSED, build_hand, compute_arc_probes, compute_link_frame, get_face_axis_for, get_link_mesh


def report_probe_quality():
    environment, entity = build_hand()
    print(f"{'link':24s} {'N':>3s} {'off-surface mm':>15s} {'|n|':>6s} {'outward':>8s} {'band°':>7s}")
    print("-" * 72)
    worst_distance = 0.0
    for name in SENSED:
        link = entity.get_link(name)
        positions, normals = compute_arc_probes(link)
        mesh = get_link_mesh(link)
        distances = trimesh.proximity.closest_point(mesh, positions)[1]
        worst_distance = max(worst_distance, float(distances.max()))

        frame = compute_link_frame(mesh, get_face_axis_for(name))
        radial = positions - frame["centre"]
        radial[:, frame["la"]] = 0.0
        outward = float((np.einsum("ij,ij->i", normals, radial) > 0).mean())
        angles = np.degrees(np.arctan2(
            positions[:, frame["v"]] - frame["centre"][frame["v"]],
            (positions[:, frame["u"]] - frame["centre"][frame["u"]]) * frame["sign"],
        ))
        band = float(angles.max() - angles.min())
        print(f"{name:24s} {len(positions):3d} {distances.max() * 1000:15.3f} "
              f"{np.linalg.norm(normals, axis=1).mean():6.3f} {outward * 100:7.0f}% {band:7.0f}")

    print("-" * 72)
    print(f"worst off-surface distance across all links: {worst_distance * 1000:.3f} mm")
    environment.close()


report_probe_quality()
