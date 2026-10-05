"""The gate's shared numeric/descriptor helpers (one copy — was verbatim ×4).

Quaternion maths, the Genesis joint-type labelling, the single-DOF "opened"
reference-joint filter, the finite-midpoint clamp, the ``GenesisWorld`` duck
stub and the CUDA thumbnail render.  The four packagers imported their own
copies of these (one drifted: artvip's ``_gs_joint_type_name`` docstring);
the shared definitions below are those bodies verbatim, with the artvip
docstring (the longer of the two, behaviour identical) attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """(w, x, y, z) quaternion → 3×3 rotation matrix."""
    w, x, y, z = (float(v) for v in quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def gs_joint_type_name(joint_type) -> str:
    """Genesis joint-type enum → a stable label (``JOINT_TYPE.REVOLUTE`` →
    ``"REVOLUTE"``); unknown types stringify their value."""
    import genesis as gs

    labels = {
        gs.JOINT_TYPE.REVOLUTE: "REVOLUTE",
        gs.JOINT_TYPE.PRISMATIC: "PRISMATIC",
        gs.JOINT_TYPE.FIXED: "FIXED",
        gs.JOINT_TYPE.FREE: "FREE",
        gs.JOINT_TYPE.SPHERICAL: "SPHERICAL",
    }
    return labels.get(joint_type, str(joint_type).split(".")[-1])


#: Backwards-compatible alias (the pre-refactor copies were underscore-private;
#: other modules/tests still reach for either spelling).
_gs_joint_type_name = gs_joint_type_name


def opened_reference_joints(joint_records: list) -> list:
    """The subset of the moving joints the opened cross-check can reference.

    Only **single-DOF** joints (the REVOLUTE/PRISMATIC ones, ``n_dofs == 1``)
    take part: upstream authors some joints as a generic ``UsdPhysics.Joint``
    which Genesis maps onto a 3-DOF SPHERICAL (its halfway midpoint is a
    3-vector — ``float(...item())`` would raise), and the offline FK walker
    (``usd_link_meshes_at_config``) only models revolute/prismatic
    displacements.  A skipped joint keeps the authored (closed) pose on BOTH
    sides of the cross-check, so the comparison stays meaningful.
    """
    return [record for record in joint_records if int(record["n_dofs"]) == 1]


#: Backwards-compatible alias (the pre-refactor copies were underscore-private).
_opened_reference_joints = opened_reference_joints


def finite_midpoint(q_min, q_max):
    """Halfway-opened position of one DOF, clamped to a finite value.

    ``0.5 * (q_min + q_max)`` is inf/NaN when a limit is ±inf (unbounded
    DOFs — Stovetop015's knobs): feeding it to ``set_dofs_position`` sends
    ``math.sin(inf)``/NaN into the FK.  A non-finite midpoint is replaced by
    the **closed** pose (0.0) — an unbounded DOF has no meaningful "halfway
    open", and the opened ``reference_qpos`` must stay finite so the bbox
    comparison never silently passes on NaN.
    """
    midpoint = 0.5 * (q_min + q_max)
    values = (
        midpoint.detach().cpu().numpy()
        if hasattr(midpoint, "detach")
        else np.asarray(midpoint)
    )
    if not np.isfinite(values).all():
        # ``inf * 0.0`` is NaN, so build the zero tensor instead of scaling.
        if hasattr(midpoint, "detach"):
            import torch

            return torch.zeros_like(midpoint.detach())
        return np.zeros_like(np.asarray(midpoint))
    return midpoint


#: Backwards-compatible alias (the pre-refactor copies were underscore-private).
_finite_midpoint = finite_midpoint


class _WorldStub:
    """Minimal duck-typed ``GenesisWorld`` stand-in for ``load_into_scene``."""

    def __init__(self, scene) -> None:
        self.scene = scene


def _render_thumbnail(descriptor, object_id: str, render_dir: Path) -> str:
    """Render one standing thumbnail of *object_id* into *render_dir*.

    The RoboCasa/ArtVIP gate render block: a SECOND throwaway scene — never
    the gate scene — placing the object on a ground plane with its lowest
    standing-pose corner at z = 0, framed from a shallow 3/4 offset camera
    (mostly horizontal so tall objects stay in frame), oriented with the
    catalog's ``standing_quaternion``.  Genesis must be initialised with the
    CUDA backend before calling (the Rasterizer camera does not work on
    CPU); a failure raises past the gate's per-object render guard.
    """
    import genesis as gs

    local = descriptor.get_local_bounding_box()
    low = local.low.detach().cpu().numpy().astype(float)
    high = local.high.detach().cpu().numpy().astype(float)

    render_dir.mkdir(parents=True, exist_ok=True)
    render_scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        standing = descriptor.standing_quaternion or (1.0, 0.0, 0.0, 0.0)
        if not np.allclose(standing[:4], (1.0, 0.0, 0.0, 0.0)):
            # Non-identity standing orientation: rotate the bbox corners into
            # the standing frame for the framing math.
            corners = np.array([
                [sx, sy, sz]
                for sx in (low[0], high[0])
                for sy in (low[1], high[1])
                for sz in (low[2], high[2])
            ]) @ quat_to_matrix(standing).T
            low, high = corners.min(axis=0), corners.max(axis=0)
        z_offset = float(-low[2])  # lowest corner sits on the plane
        center = ((low + high) / 2.0).copy()
        center[2] += z_offset  # the camera frames the *placed* object
        radius = max(max(hi - lo, 1e-3) for lo, hi in zip(low, high)) / 2.0

        render_scene.add_entity(gs.morphs.Plane())
        descriptor.load_into_scene(
            _WorldStub(render_scene),
            pos=(0.0, 0.0, z_offset),
            quat=standing,
        )
        offset = np.array([1.0, -1.0, 0.6])
        offset = offset / np.linalg.norm(offset) * (2.2 * radius)
        camera = render_scene.add_camera(
            res=(480, 480),
            pos=tuple(center + offset),
            lookat=(center[0], center[1], center[2]),
            fov=50,
        )
        render_scene.build(n_envs=0)
        rgb = camera.render(rgb=True)
        if isinstance(rgb, tuple):
            rgb = rgb[0]
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        out_file = render_dir / f"{object_id}_standing.png"
        import imageio

        imageio.imwrite(out_file, np.asarray(rgb)[..., :3])
        return str(out_file)
    finally:
        try:
            render_scene.destroy()
        except Exception:  # noqa: BLE001 — teardown is best-effort
            pass
