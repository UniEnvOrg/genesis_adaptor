"""Shared setup for the tactile-layout viz scripts; geometry only, so CPU is enough."""

from __future__ import annotations

import os

import numpy as np
import genesis as gs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

SIDE = "right"

# Kept in step with TactileSensorNodeOptions; override per-run by editing here.
FACE_AXIS = "+x"          # the default; the thumb's per-link overrides come from the collection
ARC = 120.0
N_AROUND, N_ALONG = 3, 4
MARGIN = 0.12

SENSED = [f"{SIDE}_wuji_palm"] + [
    f"{SIDE}_finger{f}_link{s}" for f in range(1, 6) for s in (2, 3, 4)
]  # palm plate + three segments on each finger = 16 links
PLAIN = [f"{SIDE}_finger{f}_link1" for f in range(1, 6)]  # no collision geometry


def out_path(name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def build_hand():
    """A built scene holding the Tianji robot; returns ``(env, entity)`` for the caller to close."""
    # gs.init() is the caller's responsibility throughout this repo, and a
    # script is the caller, so it happens here rather than in library code.
    if not gs._initialized:
        from genesis.constants import backend as gs_backend

        gs.init(backend=gs_backend.cpu)

    from unienv_interface.world import WorldEnv
    from unienv_genesis import GenesisWorld
    from unienv_genesis_collection.robots.tianji_marvin_wuji import TianjiMarvinWuji

    world = GenesisWorld()
    robot = TianjiMarvinWuji(world, name="tianji")
    env = WorldEnv(world, [robot], render_mode=None)
    env.reset(seed=0, reload=True)
    return env, world.scene.get_entity("tianji")


def link_pose(link):
    """The link's world ``(position, quaternion)``. Valid only once the scene is built."""
    return (
        np.asarray(link.get_pos().cpu()).reshape(3),
        np.asarray(link.get_quat().cpu()).reshape(4),
    )


def world_T(link) -> np.ndarray:
    """The link's world transform. Valid only once the scene is built."""
    from genesis.utils.geom import trans_quat_to_T

    return trans_quat_to_T(*link_pose(link))


def to_world(points, link) -> np.ndarray:
    """Link-frame points -> world."""
    from genesis.utils.geom import transform_by_trans_quat

    return transform_by_trans_quat(np.asarray(points, float), *link_pose(link))


def to_link(points, link) -> np.ndarray:
    """World points -> ``link``'s frame."""
    from genesis.utils.geom import inv_transform_by_trans_quat

    return inv_transform_by_trans_quat(np.asarray(points, float), *link_pose(link))


def view_matrix(eye, target, up) -> np.ndarray:
    """World -> camera, i.e. the inverse of the pose Genesis would give that camera."""
    from genesis.utils.geom import pos_lookat_up_to_T

    return np.linalg.inv(
        pos_lookat_up_to_T(
            np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
        )
    )


def face_axes() -> dict:
    """The sensor's ``{"+x": (axis, sign), ...}`` table."""
    from unienv_genesis.nodes.sensors.tactile_node import _FACE_AXES

    return _FACE_AXES


def link_mesh(link):
    """The link's collision geometry in link frame, as the sensor sees it."""
    from unienv_genesis.nodes.sensors.tactile_node import _link_mesh

    return _link_mesh(link)


def face_axis_for(link_name: str) -> str:
    """The face axis the sensor actually uses for this link."""
    # Read from make_wuji_hand_tactile's own table rather than copied, so the
    # drawings cannot drift from the placement; it corrects the rotated thumb.
    from unienv_genesis_collection.robots.wuji_tactile import _THUMB_FACE_AXES

    overrides = {
        f"{side}_{link}": axis
        for side, axes in _THUMB_FACE_AXES.items()
        for link, axis in axes.items()
    }
    return overrides.get(link_name, FACE_AXIS)


def arc_probes(link, face_axis=None, arc=ARC, n_around=N_AROUND,
               n_along=N_ALONG, margin=MARGIN):
    """Probe positions and outward normals in link frame, placed as the sensor places them."""
    # face_axis defaults to whatever the sensor uses for this link.
    from unienv_genesis.nodes.sensors.tactile_node import _arc_probes

    if face_axis is None:
        face_axis = face_axis_for(getattr(link, "name", ""))
    return _arc_probes(link, face_axis, arc, n_around, n_along, margin)


def link_frame(mesh, face_axis=FACE_AXIS) -> dict:
    """The link's long axis, its cross axes ordered ``(face axis, other)``, and the AABB."""
    # The same decomposition _arc_probes uses, so anything measured or drawn
    # here lands in the coordinates the probes were placed in. Pass the link's
    # own face_axis_for(name) -- the thumb does not use the default.
    lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
    long_axis = int(np.argmax(hi - lo))
    axis, sign = face_axes()[face_axis]
    u, v = [k for k in range(3) if k != long_axis]
    if u != axis:
        u, v = v, u
    return dict(la=long_axis, u=u, v=v, sign=sign,
                lo=lo, hi=hi, centre=0.5 * (lo + hi))
