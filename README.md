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

```
priority 100 — static scene geometry (planes, tables)
priority  50 — robots and articulated entities
priority   0 — controllers and sensors that depend on entities
```

This means a floor plane is always added to the scene before a robot, and a camera that is attached to a robot link is always attached after the robot entity exists.

### Composing a Scene

```python
world  = GenesisWorld()
plane  = PlaneBuilder(world, name="floor")
table  = TableBuilder(world, name="table", pos=(0, 0, 0.6))
franka = FrankaFR3(world,   name="franka", pos=(-0.7, 0, 0.6))
camera = CameraSensorNode(world, name="camera", ...)

env = WorldEnv(world, [plane, table, franka, camera])
context, obs, info = env.reset(seed=0, reload=True)
# obs = {
#   "franka": {"joint_position": ..., "eef_position": ..., ...},
#   "camera": {"rgb": ..., "depth": ...},
# }
```

`WorldEnv` calls each node's lifecycle methods in the correct priority order automatically. You do not manage dependencies by hand.

---

## Installation

You can install the adaptor locally (not published on PyPI yet):

```bash
git clone https://github.com/UniEnvOrg/genesis_adaptor/
cd genesis_adaptor
pip install -e .
```
