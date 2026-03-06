"""Example: Franka FR3 + Hand grasping and lifting on a cluttered table.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple, TYPE_CHECKING

import torch

import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_interface.world import WorldEnv
from unienv_interface.wrapper.video_record import EpisodeVideoWrapper, EpisodeRenderStackWrapper
from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import (
    CameraSensorNode,
    CameraSensorNodeOptions,
)
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_genesis_collection.scenes.cluttered_table_manager import (
    ClutteredTableManager,
)
from unienv_genesis_collection.robots.franka_fr3_hand import FrankaFR3WithHand
from unienv_genesis_collection.tasks.tabletop.pickplace import ObjectRelocateTask
from unienv_genesis_collection.objects.base_object import GraspCandidates
from unienv_genesis_collection.objects import BoxObject, SphereObject, CylinderObject
from unienv_genesis_refpolicy.implementations.grasp_policy import BoxGripperGraspPolicy
from unienv_genesis_refpolicy.implementations.goto_policy import GoToRefPolicy
from unienv_genesis_refpolicy.implementations.repeat_action_policy import RepeatActionRefPolicy
from unienv_genesis_refpolicy.implementations.combined_policy import CombinedRefPolicy
from unienv_genesis_refpolicy.implementations.graph_policy import GraphRefPolicy

import numpy as np

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos_cluttered_fr3_hand")

TABLE_SURFACE_Z = 0.6
TABLE_SIZE = (1.0, 0.6)
TABLE_POS = (0.0, 0.0, TABLE_SURFACE_Z)
TABLE_YAW = math.pi / 2
# FR3 base sits on the table surface, offset from the centre.
ROBOT_POS = (-0.45, 0.0, TABLE_SURFACE_Z)
TOTAL_STEPS = 350
N_EPISODES = 2

def init_env():
    world = GenesisWorld(
        scene_init_kwargs={
            "sim_options": gs.options.SimOptions(substeps=5),
            "rigid_options": gs.options.RigidOptions(box_box_detection=True),
        },
    )

    # -- Floor --
    plane = PlaneBuilder(world, name="floor", height=0.0)

    # -- Table --
    table = TableBuilder(
        world,
        name="table",
        pos=TABLE_POS,
        yaw=TABLE_YAW,
    )

    # -- Robot --
    franka = FrankaFR3WithHand(
        world,
        name="franka",
        pos=ROBOT_POS,
        quat=(1, 0, 0, 0),
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

    # -- Camera -- facing the robot (front)
    cam_pos = (1.0, 0.0, 1.4)
    cam_lookat = (0.0, 0.0, 0.6)
    camera = CameraSensorNode(
        world,
        name="camera",
        option=CameraSensorNodeOptions(
            pos=cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
            res=(640, 480),
            fov=60,
        ),
        render_rgb=True,
    )

    # -- Side camera -- perpendicular view for lift verification
    side_cam_pos = (0.0, 1.2, 1.0)
    side_cam_lookat = (0.0, 0.0, 0.6)
    side_camera = CameraSensorNode(
        world,
        name="side_camera",
        option=CameraSensorNodeOptions(
            pos=side_cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(side_cam_pos, side_cam_lookat),
            res=(640, 480),
            fov=60,
        ),
        render_rgb=True,
    )

    # -- Environment --
    env = WorldEnv(
        world,
        [plane, table, franka, task, clutter, camera, side_camera],
        render_mode="auto",
    )
    return env

def init_ref_policy(env : WorldEnv):
    def top_down_filter(candidates : GraspCandidates):
        """Keep only grasps that approach from above (EEF approach-axis points down).
        """
        return candidates.approach_direction[..., 2] < 0

    GRIPPER_OPEN = torch.tensor([-1.0])
    GRIPPER_CLOSE = torch.tensor([1.0])
    world : GenesisWorld = env.get_wrapper_attr("world")
    franka : FrankaFR3WithHand = env.get_nodes_by_type(FrankaFR3WithHand)[0]
    task : ObjectRelocateTask = env.get_nodes_by_type(ObjectRelocateTask)[0]
    target_cube : BoxObject = task.target_object

    arm_eef = franka.controller["arm"]
    gripper = franka.controller["gripper"]

    def construct_grasp_policy():
        return BoxGripperGraspPolicy(
            world=world,
            eef_controller=arm_eef,
            gripper_controller=gripper,
            target_object=target_cube,
            target_entity=task.target_object_entity,
            eef_name="arm",
            gripper_name="gripper",
            filter_candidate_fn=top_down_filter,
        )
    
    def construct_lift_policy():
        eef_pos, eef_quat = arm_eef.get_eef_pose_absolute()
        lift_pos = eef_pos.detach().clone()
        lift_pos[..., 2] += 0.15  # lift 15 cm
        return CombinedRefPolicy(
            world,
            {
                "arm": GoToRefPolicy(
                    world, arm_eef, lift_pos, eef_quat.detach().clone(),
                ),
                "gripper": RepeatActionRefPolicy(world, GRIPPER_CLOSE, steps=0),
            },
            success_keys=["arm"],
        )
    
    graph = GraphRefPolicy(
        world=world,
        nodes={"grasp": construct_grasp_policy, "lift": construct_lift_policy},
        success_edges=[("grasp", "lift")],
        fail_edges=[],
        initial="grasp",
    )
    return graph

def main():
    env = init_env()
    policy = init_ref_policy(env)
    env = EpisodeVideoWrapper(env, store_dir=VIDEO_DIR, format='mp4')

    for episode in range(N_EPISODES):
        context, obs, info = env.reset(seed=episode, reload=True)
        policy.reset(seed=episode)

        for step in range(TOTAL_STEPS):
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            policy.step()

            if policy.get_success():
                print(f"Grasp + lift succeeded at step {step}!")
                break
            if policy.get_failure():
                print(f"Grasp failed at step {step}.")
                break

    env.close()

if __name__ == "__main__":
    main()
