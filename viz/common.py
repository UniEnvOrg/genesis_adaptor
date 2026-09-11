from pathlib import Path

import genesis as gs
import numpy as np
from genesis.utils.geom import inv_transform_by_trans_quat, pos_lookat_up_to_T
from genesis.utils.geom import trans_quat_to_T, transform_by_trans_quat
from unienv_genesis.nodes.sensors.tactile_node import TactileSensorNodeOptions, _build_link_mesh
from unienv_genesis.nodes.sensors.tactile_node import _compute_arc_probes, _FACE_AXES
from unienv_genesis_collection.robots.wuji_tactile import _THUMB_FACE_AXES

OUTPUT_DIR = Path(__file__).resolve().parent / "out"
SIDE = "right"
_DEFAULTS = TactileSensorNodeOptions(attaches=[])
FACE_AXIS, ARC = _DEFAULTS.face_axis, _DEFAULTS.arc_degrees
N_AROUND, N_ALONG = _DEFAULTS.n_around, _DEFAULTS.n_along
MARGIN, PROBE_RADIUS = _DEFAULTS.margin, _DEFAULTS.probe_radius

SENSED = [f"{SIDE}_wuji_palm"] + [
    f"{SIDE}_finger{finger}_link{segment}"
    for finger in range(1, 6)
    for segment in (2, 3, 4)
]
PLAIN = [f"{SIDE}_finger{finger}_link1" for finger in range(1, 6)]
_FACE_AXIS_OVERRIDES = {f"{side}_{link}": axis for side, axes in _THUMB_FACE_AXES.items()
                        for link, axis in axes.items()}


def get_output_path(name: str) -> str:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return str(OUTPUT_DIR / name)


def build_hand(with_tactile=False):
    if not gs._initialized:
        from genesis.constants import backend as gs_backend

        gs.init(backend=gs_backend.cpu)

    from unienv_interface.world import WorldEnv
    from unienv_genesis import GenesisWorld
    from unienv_genesis_collection.robots.tianji_marvin_wuji import TianjiMarvinWuji

    world = GenesisWorld()
    robot = TianjiMarvinWuji(world, name="tianji")
    actors = [robot]
    if with_tactile:
        from unienv_genesis_collection.robots.wuji_tactile import make_wuji_hand_tactile

        actors.append(make_wuji_hand_tactile(world, robot_name="tianji", sides=(SIDE,)))
    environment = WorldEnv(world, actors, render_mode=None)
    environment.reset(seed=0, reload=True)
    return environment, actors[-1] if with_tactile else world.scene.get_entity("tianji")


def ensure_fk_ready(entity):
    """Genesis 1.3 allocates the scratch fields ``forward_kinematics`` uses only on the first
    IK call; run a one-iteration IK at the current pose so FK works before the env has stepped."""
    if getattr(entity, "_IK_qpos_orig", None) is None:
        link = entity.base_link
        entity.inverse_kinematics(link, pos=link.get_pos(), quat=link.get_quat(), max_solver_iters=1)


def get_link_pose(link):
    return np.asarray(link.get_pos().cpu()).reshape(3), np.asarray(link.get_quat().cpu()).reshape(4)


def get_world_transform(link) -> np.ndarray:
    return trans_quat_to_T(*get_link_pose(link))


def transform_to_world(points, link) -> np.ndarray:
    return transform_by_trans_quat(np.asarray(points, float), *get_link_pose(link))


def transform_to_link(points, link) -> np.ndarray:
    return inv_transform_by_trans_quat(np.asarray(points, float), *get_link_pose(link))


def compute_view_matrix(eye, target, up) -> np.ndarray:
    camera_pose = pos_lookat_up_to_T(np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float))
    return np.linalg.inv(camera_pose)


def get_link_mesh(link):
    return _build_link_mesh(link)


def get_face_axis_for(link_name: str) -> str:
    return _FACE_AXIS_OVERRIDES.get(link_name, FACE_AXIS)


def compute_arc_probes(link, face_axis=None, arc=ARC, n_around=N_AROUND, n_along=N_ALONG, margin=MARGIN):
    if face_axis is None:
        face_axis = get_face_axis_for(getattr(link, "name", ""))
    return _compute_arc_probes(link, face_axis, arc, n_around, n_along, margin)


def compute_link_frame(mesh, face_axis=FACE_AXIS) -> dict:
    lower, upper = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
    long_axis = int(np.argmax(upper - lower))
    face_axis_index, face_sign = _FACE_AXES[face_axis]
    face_axis_index, cross_axis = [axis for axis in range(3) if axis != long_axis]
    if face_axis_index != _FACE_AXES[face_axis][0]:
        face_axis_index, cross_axis = cross_axis, face_axis_index
    return {
        "la": long_axis,
        "u": face_axis_index,
        "v": cross_axis,
        "sign": face_sign,
        "lo": lower,
        "hi": upper,
        "centre": 0.5 * (lower + upper),
    }
