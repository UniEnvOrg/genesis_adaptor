# UniEnv Genesis Lab

This repository contains the implementation of the adaptor that connects the Genesis simulation engine to the UniEnv's node system interface. It is designed as a **full simulation framework** in the spirit of [mjlab](https://github.com/mujocolab/mjlab) and [IsaacLab](https://isaac-sim.github.io/IsaacLab/), bringing the same modular, composable scene-building philosophy to Genesis.

## Node System

mjlab (MuJoCo's Python interface) and IsaacLab (NVIDIA's GPU-accelerated RL framework for Omniverse) both share a common idea: a simulation environment should be assembled from **independent, reusable building blocks** — robots, sensors, scene elements — each responsible for its own lifecycle. `genesis_adaptor` mirrors this philosophy for the Genesis engine, providing:

- A **node-based scene description** system that lets you compose robots, sensors, and scene geometry declaratively.
- A **priority-ordered lifecycle** that guarantees correct initialisation order without manual dependency management.
- **Pluggable controllers** that decouple the physics entity from the control law, just as IsaacLab separates actuator models from robot assets.
- **Transparent batching** so that the same node code runs identically on a single environment or thousands of parallelised GPU environments.

### Lifecycle Phases & Priorities

Each lifecycle phase is executed by sorting all registered nodes by their declared **priority** for that phase and calling them in descending order. A higher priority number runs first.

| Phase | Purpose |
|---|---|
| `reload` | Destroy the old scene; add geometry / entities to the new one. |
| `after_reload` | Scene is compiled. Initialise controllers, cache entity handles. |
| `reset` | Move entities back to their starting poses; clear queued actions. |
| `after_reset` | Populate the first observation cache after reset. |
| `pre_environment_step` | Apply the pending action to the physics engine. |
| `post_environment_step` | Read back state and refresh the observation cache. |

Typical priority assignments:

|Priority Range|Node Catgories|
|-|-|
|`[80, 100)`|static scene geometry (planes, tables)|
|`[50, 80)`|robots and articulated entities|
|`(-20, 0]`|controllers and sensors that depend on entities|
|$\mathbb{Z}$|Task control logic can declare any priority to ensure it runs after all the necessary state has been updated by the controllers and sensors.|

This means a floor plane is always added to the scene before a robot, and a camera that is attached to a robot link is always attached after the robot entity exists.


### Composing a Scene

```python
world  = GenesisWorld()
plane  = PlaneBuilder(world, name="floor")
table  = TableBuilder(world, name="table", pos=(0, 0, 0.6))
franka = FrankaFR3WithHand(world, name="franka", pos=(-0.7, 0, 0.6))
camera = CameraSensorNode(world, name="camera", ...)

env = WorldEnv(world, [plane, table, franka, camera])
context, obs, info = env.reset(seed=0, reload=True)
# obs = {
#   "franka": {"joint_position": ..., "eef_position": ..., ...},
#   "camera": {"rgb": ..., "depth": ...},
# }
# Each space describes its data, e.g.
# env.observation_space["camera"]["rgb"].metadata["layout"] == "HWC"
```

`WorldEnv` calls each node's lifecycle methods in the correct priority order automatically. You do not manage dependencies by hand.

### Space Metadata

Every observation and action space the nodes declare carries unienv [space metadata](https://github.com/UniEnvOrg/UniEnvPy/blob/main/docs/concepts/spaces-and-backends.md#space-metadata) describing what the data means, so wrappers, storages and dataset exporters can find e.g. the camera images without knowing the node:

```python
from unienv_interface.space.space_utils.metadata_utils import find_spaces_by_metadata

env.observation_space["camera"]["rgb"].metadata
# {"kind": "image", "format": "rgb", "layout": "HWC", "source": "camera"}
env.observation_space["franka"]["arm"]["joint_position"].metadata["units"]
# ["rad", "rad", ...]   one entry per DOF, "m" for prismatic joints
find_spaces_by_metadata(franka.action_space, control="gripper")
# [(("gripper",), BoxSpace(...))]
```

- **Cameras**: `rgb` (`image`, `rgb` / `HWC`), `depth` (metres, `HW`), `seg` (`segmentation`, with the scene's segmentation level), `normal`, `position` (world `pointcloud`), `intrinsics_cv` / `extrinsics_cv` / `local_extrinsics_cv` (OpenCV `camera_cv` frame). `source` is the camera node's name.
- **Joint readings and targets**: per-DOF `units` (`rad` / `rad/s` / `N*m` for revolute DOFs, `m` / `m/s` / `N` for prismatic ones) and `names` from the entity's joints; root and end-effector poses state their `frame` (`world`, or `base` with a configured `base_frame`) and `rotation="quat_wxyz"`.
- **Actions**: `control` / `mode` (`joint_position`, `joint_velocity`, `eef_pose` with the `rotation` of its `rotation_representation`, `gripper` for a mimic gripper with `open_value` / `closed_value`).
- **Tasks**: `kind="task"` goals and targets in the world frame.

Controller configs are checked as well: a keyword argument that no controller uses raises an `UnknownControllerOptionWarning` (see `EntityController.extra_attributes` for the extra attributes a controller config may carry).

### Articulated Rooms

Rooms furnished with **fixed articulated furniture** (RoboCasa kitchen cabinets,
drawers and appliances; ReplicaCAD fridges, cupboards and doors; ArtVIP USD
appliances; MolmoSpaces Ai2-THOR furniture; Lightwheel SimReady assets; NVIDIA
SimReady Warehouse rigid props) are built by
`RoomBuilder` (baked layouts or procedural sampling), and a single named joint of
one furniture instance can be driven by `OpenArticulatedJointTask` — dense
fraction-tracking reward, per-env goals, masked resets:

```python
room = RoomBuilder(world, name="room", layout_yaml="robocasa_kitchen/open_kitchen")
task = OpenArticulatedJointTask(
    world, room, furniture_key="cabinet_hinge", joint_name="rightdoorhinge",
    grasp_link_name="cabinet_hinge_right_door_handle",
)
env = WorldEnv(world, [room, franka, task])
```

Procedural rooms sample from a catalog-backed furniture pool, e.g.
`make_artvip_pool(tags="kitchen", n=6)` for the ArtVIP USD appliances,
`make_molmospaces_pool(tags="kitchen", n=6)` for the MolmoSpaces THOR
furniture, `make_lightwheel_pool(tags="kitchen", n=6)` for the Lightwheel
SimReady assets or `make_simready_warehouse_pool(tags="prop", n=8)` for the
Warehouse rigid props (see
[docs/articulated_rooms.md](docs/articulated_rooms.md)).

See [docs/articulated_rooms.md](docs/articulated_rooms.md) for the API, the
packaged assets (RoboCasa CC BY 4.0; **ArtVIP is Apache-2.0 per its card with no
committed LICENSE file — caveat there**; **ReplicaCAD and Lightwheel are
CC BY-NC 4.0 — non-commercial**, see the license caveats there; MolmoSpaces is
CC BY 4.0 with attribution; **the SimReady Warehouse card tags CC BY 4.0 but the
archive commits no LICENSE file — caveat there**) and how to regenerate them,
plus
the runnable
[`examples/room_open_drawer_fr3.py`](examples/room_open_drawer_fr3.py).

---

## Installation

Install from PyPI:

```bash
pip install unienv_genesis
```

Or install the adaptor locally from source:

```bash
git clone https://github.com/UniEnvOrg/genesis_adaptor/
cd genesis_adaptor
pip install -e .
```
