import numpy as np
import trimesh

from common import (ARC, N_ALONG, N_AROUND, PROBE_RADIUS, SENSED, build_hand,
                    compute_arc_probes, compute_link_frame, get_face_axis_for, get_link_mesh)

SAMPLE_COUNT = 120_000
SLAB_HALF_WIDTH = 0.004


def measure_sensed_band(mesh, probes, radius, face_axis):
    frame = compute_link_frame(mesh, face_axis)
    long_axis, face_axis_index = frame["la"], frame["u"]
    cross_axis, centre = frame["v"], frame["centre"]
    rows = np.unique(np.round(probes[:, long_axis], 6))
    points, _ = trimesh.sample.sample_surface(mesh, SAMPLE_COUNT)
    band = points[np.abs(points[:, long_axis] - rows[len(rows) // 2]) < SLAB_HALF_WIDTH]
    if len(band) < 50:
        return float("nan")
    angles = np.degrees(np.arctan2(band[:, cross_axis] - centre[cross_axis],
                                   band[:, face_axis_index] - centre[face_axis_index]))
    covered = np.linalg.norm(band[:, None, :] - probes[None, :, :], axis=2).min(axis=1) <= radius
    if not covered.any():
        return 0.0
    covered_angles = np.sort(angles[covered])
    gaps = np.diff(np.concatenate([covered_angles, [covered_angles[0] + 360]]))
    return float(360 - gaps.max())


def report_sensed_bands():
    environment, entity = build_hand()
    print(f"{N_AROUND}x{N_ALONG} taxels, arc_degrees={ARC:.0f}, probe_radius={PROBE_RADIUS * 1000:.0f} mm")
    print(f"{'link':24s} {'sensed band':>12s}")
    print("-" * 38)
    bands = []
    for name in SENSED:
        link = entity.get_link(name)
        probes, _ = compute_arc_probes(link)
        band = measure_sensed_band(get_link_mesh(link), probes, PROBE_RADIUS, get_face_axis_for(name))
        bands.append(band)
        print(f"{name:24s} {band:11.0f}°")

    fingers = np.array(bands[1:])
    print("-" * 38)
    print(f"finger segments {fingers.min():.0f}-{fingers.max():.0f}° "
          f"(mean {fingers.mean():.0f}°), palm {bands[0]:.0f}°")
    environment.close()


report_sensed_bands()
