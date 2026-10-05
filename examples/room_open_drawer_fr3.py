"""Example: open a kitchen cabinet door with Franka FR3 + Robotiq 2F-85 in a room.

End-to-end wiring of the articulated-furniture stack (plan stages S1-S7):

* :class:`RoomBuilder` builds a room — a baked RoboCasa kitchen shell plus its
  fixed articulated fixtures (``assets/scenes/rooms/robocasa_kitchen/``).
* :class:`FrankaFR3WithRobotiq2F85` spawns on the room floor *in front of the
  cabinet*: the spawn is derived from the baked layout (the cabinet's
  room-frame front approach point, mapped to the world frame through
  :meth:`RoomBuilder.placement_world_pose`) and faces the fixture.
* :class:`OpenArticulatedJointTask` binds to **one named joint of one
  furniture instance** (``cabinet_hinge`` / ``rightdoorhinge``) and exposes a
  dense fraction-tracking reward (``-|fraction - goal|``) with a randomized
  goal — the interface a real manipulation policy would consume.
* A ``GoToRefPolicy`` drives the hand towards a pre-grasp point in front of the
  door handle.  The point of this example is the *wiring* (scene, robot, task,
  observations), not a solved manipulation policy: the reach is open-loop.

The example targets the cabinet's right hinge (fraction ``0`` = closed, ``1``
= fully open; the goal is sampled inside the physically clean opening band,
see ``TARGET_FRACTION_RANGE``).  The packaged RoboCasa ``drawer`` fixture is
the drop-in alternative (furniture key ``"drawer"``, joint ``"slidejoint"``,
handle ``"drawer_door_handle"``); note that its slide is authored closed at
the *upper* limit (range ``[-0.6, 0]``), so its normalised fraction runs
backwards relative to the "1 = open" intuition.

Usage::

    # GPU (default), 90 control steps, renders overhead + close-up frames
    python examples/room_open_drawer_fr3.py --backend cuda --steps 90

    # CPU, 4 parallel environments, no rendering
    python examples/room_open_drawer_fr3.py --backend cpu --n-envs 4 --no-render

    # Write the frames next to the script instead of /tmp
    python examples/room_open_drawer_fr3.py --out-dir examples/room_open_drawer_fr3_frames
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile

import genesis as gs
from genesis.constants import backend as gs_backend

# -- Scene constants -------------------------------------------------------
#: Baked layout id (``scenes/rooms/robocasa_kitchen/layouts/open_kitchen.json``).
LAYOUT_ID = "robocasa_kitchen/open_kitchen"
#: Furniture instance key inside the layout (``FurniturePlacement.object_id``).
FURNITURE_KEY = "cabinet_hinge"
#: Joint of that fixture to drive.  RoboCasa's right door hinge is authored
#: closed at its lower limit (``[0, 3]`` rad), so fraction 0 = closed and
#: fraction 1 = fully open — the task's natural direction.
JOINT_NAME = "rightdoorhinge"
#: Handle link whose world position the task reports as an observation.
HANDLE_LINK_NAME = "cabinet_hinge_right_door_handle"
#: Goal fraction range sampled per episode.  The task defaults to ``(0.6, 1.0)``,
#: but this fixture's hinge range (``0 .. 3`` rad) overshoots the cabinet body
#: above ~0.6 (103 deg): the solver then pushes the door back and the joint
#: drifts on its own.  Fractions 0.4-0.6 (1.2-1.8 rad, i.e. 69-103 deg) are
#: the physically clean opening band, and every sampled initial state (drawn
#: from ``[0, goal - success_threshold]``) is inside the stable band too.
TARGET_FRACTION_RANGE = (0.4, 0.6)
#: Distance kept between the robot base and the fixture's front face (metres).
ROBOT_STANDOFF = 1.0
#: Offset of the scripted pre-grasp point from the handle (metres): in front
#: of the door (the fixture's local ``-y``) and slightly above the handle.
PRE_GRASP_FRONT_OFFSET = 0.12
PRE_GRASP_UP_OFFSET = 0.06


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", choices=("cpu", "cuda"), default="cuda",
        help="Genesis backend to initialise (default: cuda).",
    )
    parser.add_argument(
        "--n-envs", type=int, default=1,
        help="Number of parallel environments (1 = unbatched world; default: 1).",
    )
    parser.add_argument(
        "--steps", type=int, default=90,
        help="Number of control steps to run (default: 90).",
    )
    parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=True,
        help="Add the overhead / close-up cameras and save frames (default: on).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the scene reload and the task's goal sampling (default: 0).",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=os.path.join(tempfile.gettempdir(), "room_open_drawer_fr3"),
        help="Directory the rendered frames are written to "
             "(default: <tmp>/room_open_drawer_fr3).",
    )
    args = parser.parse_args()
    if args.n_envs < 1:
        parser.error("--n-envs must be >= 1")
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    return args


def _scalar(value) -> float:
    """First element of a (possibly batched) scalar signal, as a float."""
    if hasattr(value, "reshape"):
        return float(value.reshape(-1)[0])
    return float(value)


def _summary(value) -> str:
    """Compact ``shape=[...] values=[...]`` string for tensors / scalars."""
    if hasattr(value, "reshape") and hasattr(value, "shape"):
        values = [round(float(v), 4) for v in value.reshape(-1).tolist()]
        return f"shape={tuple(value.shape)} values={values}"
    return repr(value)


def init_env(args: argparse.Namespace, torch):
    """Build the room + robot + task environment (the wiring this file shows)."""
    from unienv_interface.world import WorldEnv
    from unienv_genesis import GenesisWorld
    from unienv_genesis.nodes.sensors.camera_node import (
        CameraSensorNode,
        CameraSensorNodeOptions,
    )
    from unienv_genesis_collection.robots.franka_fr3_robotiq_2f85 import (
        FrankaFR3WithRobotiq2F85,
    )
    from unienv_genesis_collection.scenes.room_builder import (
        RoomBuilder,
        load_baked_room_layout_spec,
        yaw_to_quat,
    )
    from unienv_genesis_collection.tasks.articulated import OpenArticulatedJointTask

    world = GenesisWorld(
        # A single environment needs no batch dimension (the adaptor's
        # unbatched mode); more than one builds a batched Genesis scene.
        batch_size=None if args.n_envs == 1 else args.n_envs,
        scene_init_kwargs=dict(
            # The `--render` cameras use the classic rasterizer, whose lighting
            # comes from VisOptions.  The default directional light is blocked
            # by the room walls (the interior would render nearly black), so
            # the room is lit by a vertical light plus generous ambient.
            vis_options=gs.options.VisOptions(
                ambient_light=(0.5, 0.5, 0.5),
                lights=[
                    gs.options.vis.DirectionalLight(
                        dir=(0.0, 0.0, -1.0), color=(1.0, 1.0, 1.0), intensity=3.0,
                    ),
                ],
            ),
            sim_options=gs.options.SimOptions(dt=1 / 15, substeps=10),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=False,
        ),
    )

    # -- Room: RoboCasa kitchen shell + fixtures ---------------------------
    room = RoomBuilder(world, name="room", layout_yaml=LAYOUT_ID, seed=args.seed)

    # -- Robot: stand `ROBOT_STANDOFF` in front of the cabinet -------------
    # The baked layout is a plain JSON spec, so the robot spawn can be derived
    # before the scene is built.  Furniture fronts face the fixture's local
    # -y (the RoboCasa convention), rotated by the placement yaw; the room
    # frame origin is the floor centre, so the approach point is converted
    # with RoomBuilder.placement_world_pose.
    layout = load_baked_room_layout_spec(LAYOUT_ID, kind="layout")
    placement = next(p for p in layout.furniture if p.object_id == FURNITURE_KEY)
    front_dir = (math.sin(placement.yaw), -math.cos(placement.yaw))
    approach_point = (
        placement.pos[0] + ROBOT_STANDOFF * front_dir[0],
        placement.pos[1] + ROBOT_STANDOFF * front_dir[1],
        0.0,  # on the room floor
    )
    base_pos, _ = room.placement_world_pose(approach_point)
    # Robot forward is its local +x, so facing the fixture front (-front_dir)
    # is a yaw of `placement.yaw + pi/2`.
    base_yaw = placement.yaw + math.pi / 2
    franka = FrankaFR3WithRobotiq2F85(
        world,
        name="franka",
        pos=tuple(float(v) for v in base_pos),
        quat=yaw_to_quat(base_yaw),
    )

    # -- Task: open the cabinet's right door to a sampled fraction ---------
    task = OpenArticulatedJointTask(
        world,
        room_builder=room,
        furniture_key=FURNITURE_KEY,
        joint_name=JOINT_NAME,
        grasp_link_name=HANDLE_LINK_NAME,
        target_fraction_range=TARGET_FRACTION_RANGE,
        success_threshold=0.05,
        seed=args.seed,
    )

    # -- Cameras (optional): overhead room view + cabinet close-up ---------
    nodes = [room, franka, task]
    if args.render:
        overview_pos = (0.0, -2.4, 2.2)
        overview = CameraSensorNode(
            world,
            name="overview_camera",
            option=CameraSensorNodeOptions(
                backend="genesis",
                pos=overview_pos,
                quat=CameraSensorNodeOptions.quat_from_lookat(
                    overview_pos, (0.0, 1.4, 0.6),
                ),
                res=(960, 540),
                fov=60,
            ),
            render_rgb=True,
        )
        closeup_pos = (placement.pos[0] + 0.15, placement.pos[1] - 1.9, 1.0)
        closeup = CameraSensorNode(
            world,
            name="closeup_camera",
            option=CameraSensorNodeOptions(
                backend="genesis",
                pos=closeup_pos,
                quat=CameraSensorNodeOptions.quat_from_lookat(
                    closeup_pos, (placement.pos[0], placement.pos[1], 0.5),
                ),
                res=(960, 540),
                fov=60,
            ),
            render_rgb=True,
        )
        nodes += [overview, closeup]

    env = WorldEnv(world, nodes, render_mode="auto" if args.render else None)
    return env, world, room, franka, task, placement


def init_reach_policy(world, franka, task, placement):
    """Scripted reach towards a pre-grasp point in front of the door handle.

    Builds a :class:`GoToRefPolicy` from the task's own ``handle_position``
    observation (world frame), offset towards the room (the fixture's front)
    and slightly up.  The target orientation is the robot's initial
    end-effector orientation, so the hand keeps its approach angle.
    """
    import torch
    from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy

    arm = franka.controller["arm"]
    handle_position = task.get_observation()["handle_position"]
    front_dir = torch.tensor(
        [math.sin(placement.yaw), -math.cos(placement.yaw), 0.0],
        dtype=world.dtype,
        device=world.device,
    )
    pre_grasp = (
        handle_position
        + PRE_GRASP_FRONT_OFFSET * front_dir
        + torch.tensor(
            [0.0, 0.0, PRE_GRASP_UP_OFFSET], dtype=world.dtype, device=world.device,
        )
    )
    _, eef_quat = arm.get_eef_pose_absolute()
    return GoToRefPolicy(world, arm, pre_grasp, eef_quat.detach().clone())


def _save_frame(frame, out_dir: str, tag: str) -> str:
    """Save one rendered frame (tensor or numpy, ``(H, W, 3)`` uint8) as PNG."""
    import numpy as np
    from PIL import Image

    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:  # (n_envs, H, W, 3) — keep the first environment
        frame = frame[0]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"frame_{tag}.png")
    Image.fromarray(frame.astype(np.uint8)).save(path)
    return path


def main() -> None:
    args = _parse_args()

    # The collection / adaptor modules import `genesis` at import time and
    # must not be imported before the backend is initialised (the repo-wide
    # convention used by every example and test module).
    gs.init(backend=gs_backend.gpu if args.backend == "cuda" else gs_backend.cpu)

    import torch

    from unienv_genesis.utils.batch_dim_utils import get_batch_aware_shape
    from unienv_genesis_collection.objects.articulated_object import (
        resolve_joint_dofs,
        resolve_link_index,
    )

    env, world, room, franka, task, placement = init_env(args, torch)
    print(f"Backend: {args.backend} | device: {world.device} | "
          f"batch_size: {world.batch_size}")

    context, obs, info = env.reset(seed=args.seed, reload=True)
    print(f"Reset ok — node observation keys: {sorted(obs)}")
    print(f"Room: layout={LAYOUT_ID!r} size={room.room_size}")
    print(f"Furniture keys: {sorted(room.furniture)}")

    # Runtime name resolution on the *built* entity (the path the task uses):
    # fixture ids, joint and handle link names come from the packaged assets,
    # and the fraction limits are read from the compiled scene.
    entity = room.furniture_entities[FURNITURE_KEY]
    dofs_idx = resolve_joint_dofs(entity, JOINT_NAME)
    resolve_link_index(entity, HANDLE_LINK_NAME)
    q_min, q_max = entity.get_dofs_limit(dofs_idx)
    print(
        f"Furniture {FURNITURE_KEY!r}: joint {JOINT_NAME!r} "
        f"(dofs {list(dofs_idx)}, limits {_scalar(q_min):.3f}..{_scalar(q_max):.3f}), "
        f"handle link {HANDLE_LINK_NAME!r}"
    )
    print(f"Furniture placement: pos={placement.pos} yaw={placement.yaw:.3f}")
    print(f"Task info: {info.get(task.name, {})}")

    print("Initial task observation:")
    for key, value in task.get_observation().items():
        print(f"  {key}: {_summary(value)}")

    # -- Actions: scripted reach (arm) + open gripper ----------------------
    reach = init_reach_policy(world, franka, task, placement)
    arm = franka.controller["arm"]
    gripper_action = torch.full(
        get_batch_aware_shape(world.batch_size, 1),
        -1.0,  # Robotiq convention: -1 = fully open
        dtype=world.dtype,
        device=world.device,
    )

    frames = []
    reward = None
    print("Stepping:")
    for step in range(args.steps):
        action = {"arm": reach.get_action(), "gripper": gripper_action}
        obs, reward, terminated, truncated, info = env.step(action)
        reach.step(dt=info["control_step_elapsed_time"])

        task_obs = task.get_observation()
        if step == 0 or (step + 1) % 15 == 0 or step == args.steps - 1:
            eef_pos = arm.get_eef_pose_absolute()[0]
            print(
                f"step {step + 1:4d} | fraction {_scalar(task_obs['joint_position']):.3f} "
                f"(goal {_scalar(task_obs['target_fraction']):.3f}) "
                f"| reward {_scalar(reward):+.4f} "
                f"| terminated {bool(torch.as_tensor(terminated).reshape(-1)[0])} "
                f"| eef {[round(float(v), 3) for v in eef_pos.reshape(-1).tolist()]}"
                f" | reach_done {bool(torch.as_tensor(reach.get_success()).reshape(-1)[0])}"
            )

        if args.render:
            # `env.render()` returns the camera frame directly for a single
            # renderable node, or a `{node_name: frame}` mapping for several.
            frame = env.render()
            if step == 0 or step == args.steps - 1:
                frames.append((step + 1, frame))

    # -- Summary -----------------------------------------------------------
    print("Final task observation:")
    for key, value in task.get_observation().items():
        print(f"  {key}: {_summary(value)}")
    last_reward = _scalar(reward)
    print(
        f"Last reward {last_reward:+.4f} "
        f"(finite: {math.isfinite(last_reward)}; the task's dense reward lives in [-1, 0])"
    )
    for step, frame in frames:
        if isinstance(frame, dict):  # one entry per camera node
            for name, value in frame.items():
                tag = f"step{step:04d}_{name}"
                print(f"Saved frame: {_save_frame(value, args.out_dir, tag)}")
        else:
            print(f"Saved frame: {_save_frame(frame, args.out_dir, f'step{step:04d}')}")

    env.close()


if __name__ == "__main__":
    main()
