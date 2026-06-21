"""Example: Franka FR3 + Robotiq 2F-85 pick-and-place with PickPlaceRefPolicy.

Demonstrates the full pick-and-place pipeline:
    grasp → lift → move_to_goal → lower → release → retreat

A red target cube is randomly placed on the table and must be moved to a
random goal position.  The :class:`PickPlaceRefPolicy` drives the robot
through the multi-phase sequence, and camera observations are recorded as
videos.
"""

from __future__ import annotations

import math
import os

import torch

import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.gpu)

from unienv_interface.world import WorldEnv
from unienv_interface.wrapper.video_record import EpisodeVideoWrapper
from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import (
    CameraSensorNode,
    CameraSensorNodeOptions,
)
from unienv_genesis_collection.scenes.env_map_manager import EnvMapManager
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_genesis_collection.scenes.cluttered_table_manager import (
    ClutteredTableManager,
)
from unienv_genesis_collection.robots.franka_fr3_robotiq_2f85 import (
    FrankaFR3WithRobotiq2F85,
    ZedSupportedResolution,
)
from unienv_genesis_collection.tasks.tabletop.pickplace import ObjectRelocateTask
from unienv_genesis_collection.objects import BoxObject, SphereObject, CylinderObject
from unienv_genesis_refpolicy.implementations.pickplace_policy import PickPlaceRefPolicy
from unienv_genesis_refpolicy.implementations.grasp_policy import BoxGripperGraspPolicy

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos_pickplace_fr3_robotiq")

TABLE_SURFACE_Z = 0.6
TABLE_SIZE = (1.0, 0.6)
TABLE_POS = (0.0, 0.0, TABLE_SURFACE_Z)
TABLE_YAW = math.pi / 2
ROBOT_POS = (-0.45, 0.0, TABLE_SURFACE_Z)
TOTAL_STEPS = 500
N_EPISODES = 2


def init_env():
    world = GenesisWorld(
        scene_init_kwargs=dict(
            sim_options=gs.options.SimOptions(dt=1/15, substeps=10),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
        ),
        # Use only image-based lighting from the HDR environment map.
        lights={},
    )

    # -- Environment map --
    # The HDR env map provides background and image-based lighting for the
    # Nyx cameras. EnvMapManager runs at reload priority 150, so its selection
    # is picked up when the Nyx cameras (priority 0) are (re-)created later in
    # the reload sequence.
    env_map = EnvMapManager(world, name="env_map")

    # -- Table --
    table = TableBuilder(
        world,
        name="table",
        pos=TABLE_POS,
        yaw=TABLE_YAW,
    )

    # -- Robot --
    franka = FrankaFR3WithRobotiq2F85(
        world,
        name="franka",
        pos=ROBOT_POS,
        quat=(1, 0, 0, 0),
    )

    # -- Gripper camera (DROID-style, attached to g_base) --
    franka_gripper_cam = CameraSensorNode(
        world,
        name="gripper_cam",
        option=franka.get_droid_gripper_cam_options(
            backend="nyx",
            resolution=ZedSupportedResolution.HD720,
        ),
        render_rgb=True,
    )

    # -- Distractor objects --
    distractor_pool = [
        BoxObject(size=(0.05, 0.03, 0.06), name="green_box", color=(0, 0.8, 0.2, 1)),
        SphereObject(radius=0.025, name="blue_ball", color=(0, 0.3, 1, 1)),
        SphereObject(radius=0.02, name="yellow_ball", color=(1, 1, 0, 1)),
        CylinderObject(
            radius=0.02, height=0.08, name="orange_can", color=(1, 0.5, 0, 1),
        ),
        CylinderObject(
            radius=0.015, height=0.06, name="purple_peg", color=(0.6, 0, 0.8, 1),
        ),
    ]

    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=distractor_pool,
        n_objects_range=(3, 5),
        table_size=TABLE_SIZE,
        table_pos=TABLE_POS,
        table_yaw=TABLE_YAW,
        random_quaternion=True,
    )

    # -- Task: relocate the red target cube --
    target_cube = BoxObject(
        size=(0.04, 0.04, 0.04), name="target_cube", color=(1, 0, 0, 1),
    )
    task = ObjectRelocateTask(
        world,
        table_builder=table,
        clutter_manager=clutter,
        target_object=target_cube,
        success_threshold=0.03,
    )
    # For this demo rollout, the video episode boundary should be policy
    # completion (full pick-and-place), not the task's object-at-goal signal
    # (which fires mid-sequence during the "lower" phase, splitting the video
    # before the visually interesting release/retreat phases are captured).
    task.has_termination_signal = False

    # -- Front camera -- facing the robot --
    cam_pos = (1.0, 0.0, 1.4)
    cam_lookat = (0.0, 0.0, 0.6)
    camera = CameraSensorNode(
        world,
        name="camera",
        option=CameraSensorNodeOptions(
            backend="nyx",
            pos=cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
            res=(1280, 720),
            fov=60,
        ),
        render_rgb=True,
    )

    # -- Side camera -- perpendicular view for pick-and-place verification --
    side_cam_pos = (0.0, 1.2, 1.0)
    side_cam_lookat = (0.0, 0.0, 0.6)
    side_camera = CameraSensorNode(
        world,
        name="side_camera",
        option=CameraSensorNodeOptions(
            backend="nyx",
            pos=side_cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(side_cam_pos, side_cam_lookat),
            res=(1280, 720),
            fov=60,
        ),
        render_rgb=True,
    )

    # -- Environment --
    env = WorldEnv(
        world,
        [env_map, table, franka, franka_gripper_cam, task, clutter, camera, side_camera],
        render_mode="auto",
    )
    return env


def init_ref_policy(env: WorldEnv):
    """Construct the PickPlaceRefPolicy from the env's nodes."""
    world: GenesisWorld = env.get_wrapper_attr("world")
    franka: FrankaFR3WithRobotiq2F85 = env.get_nodes_by_type(FrankaFR3WithRobotiq2F85)[0]
    task: ObjectRelocateTask = env.get_nodes_by_type(ObjectRelocateTask)[0]
    target_cube: BoxObject = task.target_object

    arm_eef = franka.controller["arm"]
    gripper = franka.controller["gripper"]

    policy = PickPlaceRefPolicy(
        world=world,
        eef_controller=arm_eef,
        gripper_controller=gripper,
        task=task,
        target_object=target_cube,
        eef_name="arm",
        gripper_name="gripper",
        lift_height=0.15,
        retreat_height=0.15,
        release_seconds=1.0,
        grasp_kwargs=dict(
            filter_candidate_fn=BoxGripperGraspPolicy.filter_fn_top_down,
        ),
    )
    return policy


def main():
    env = init_env()
    policy = init_ref_policy(env)
    env = EpisodeVideoWrapper(env, store_dir=VIDEO_DIR, format='mp4')

    for episode in range(N_EPISODES):
        context, obs, info = env.reset(seed=episode, reload=True)
        policy.reset(seed=episode)

        for step in range(TOTAL_STEPS):
            action = policy.get_action()
            obs, reward, terminated, truncated, info = env.step(action)
            policy.step()

            if policy.get_success():
                print(f"Pick-and-place succeeded at step {step}!")
                break
            if policy.get_failure():
                print(f"Pick-and-place failed at step {step}.")
                break

    env.close()


if __name__ == "__main__":
    main()
