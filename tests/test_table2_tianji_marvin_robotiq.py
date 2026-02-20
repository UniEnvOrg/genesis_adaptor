"""Test: Tianji Marvin dual-arm robot with Robotiq 2F-85 grippers."""

import math
import os
import torch
import genesis as gs
from genesis.constants import backend as gs_backend
gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import CameraSensorNode, CameraSensorNodeOptions
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_genesis_collection.scenes.goal_region_builder import GoalRegionBuilder
from unienv_genesis_collection.robots.tianji_marvin_robotiq_2f85 import TianjiMarvinWithRobotiq2F85
from unienv_interface.world import WorldEnv
from unienv_interface.wrapper.video_record import EpisodeVideoWrapper

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos_tianji_marvin_robotiq2")


def test_table_and_tianji_marvin_robotiq():
    world = GenesisWorld()

    TABLE_SURFACE_Z = 0.6
    ROBOT_Z = TABLE_SURFACE_Z + 0.35  # Raise base above table for downward reach

    plane = PlaneBuilder(world, name="floor", height=0.0)
    table = TableBuilder(
        world, name="table",
        pos=(0, 0, TABLE_SURFACE_Z),
        quat=(0.7071068, 0, 0, 0.7071068),
    )

    table2 = TableBuilder(world, name="table2", pos=(-1.0, 0.3, TABLE_SURFACE_Z))
    table3 = TableBuilder(world, name="table3", pos=(-1.0, -0.3, TABLE_SURFACE_Z))

    goal_region = GoalRegionBuilder(world, name="goal_region", pos=(0.2, 0, TABLE_SURFACE_Z + 1e-4))

    tianji = TianjiMarvinWithRobotiq2F85(
        world,
        name="tianji",
        pos=(-0.7, 0, ROBOT_Z),
        quat=(1, 0, 0, 0),
    )

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
    from xbarray.transformations.rotation_conversions.pytorch import (
        quaternion_to_matrix, matrix_to_euler_angles,
    )

    N_VIDEOS = 2
    total_steps = 350

    env = EpisodeVideoWrapper(
        WorldEnv(world, [plane, table, table2, table3, goal_region, tianji, camera], render_mode="auto"),
        store_dir=VIDEO_DIR,
        format="mp4",
    )

    for episode in range(N_VIDEOS):
        context, obs, info = env.reset(seed=episode, reload=True)
        assert obs is not None, f"Expected observation (episode {episode})"

        left_obs = obs["tianji"]["left_arm"]
        right_obs = obs["tianji"]["right_arm"]

        left_eef_pos = left_obs["eef_position"].clone()
        left_eef_quat = left_obs["eef_quaternion"].clone()
        left_euler = matrix_to_euler_angles(quaternion_to_matrix(left_eef_quat), "XYZ")

        right_eef_pos = right_obs["eef_position"].clone()
        right_eef_quat = right_obs["eef_quaternion"].clone()
        right_euler = matrix_to_euler_angles(quaternion_to_matrix(right_eef_quat), "XYZ")

        left_home = left_eef_pos.clone()
        right_home = right_eef_pos.clone()

        # Target EEF orientation: grippers pointing forward (+X).
        # Euler [0, pi/2, 0] in XYZ convention rotates EEF Z to world +X.
        left_euler_fwd = torch.tensor([0, math.pi / 2, 0], dtype=torch.float32)
        right_euler_fwd = torch.tensor([0, math.pi / 2, 0], dtype=torch.float32)

        # Forward-reach targets: arms reach forward (+X) and inward, hovering
        # above the table.  IK is stable with fwd<=0.35, inward_frac>=0.4,
        # and z >= TABLE_SURFACE_Z + 0.10.
        left_table = left_home.clone()
        left_table[0] = left_home[0] + 0.30     # +X forward (30 cm)
        left_table[1] = left_home[1] * 0.4      # fold inward (40% of home Y)
        left_table[2] = TABLE_SURFACE_Z + 0.10  # 10 cm above table
        right_table = right_home.clone()
        right_table[0] = right_home[0] + 0.30
        right_table[1] = right_home[1] * 0.4
        right_table[2] = TABLE_SURFACE_Z + 0.10

        OPEN = torch.tensor([-1.0])
        CLOSE = torch.tensor([1.0])

        # 3-phase dual-arm demo — reach forward over the table:
        #   Phase 0 (  0-149): Both arms reach forward, inward, and down
        #                      while rotating grippers to face +X
        #   Phase 1 (150-249): Hold above table, close grippers
        #   Phase 2 (250-349): Return home, open grippers, restore orientation
        for step in range(total_steps):
            if step < 150:
                # Phase 0: reach forward + inward + lower + rotate grippers
                alpha = step / 150.0
                left_target = (1 - alpha) * left_home + alpha * left_table
                right_target = (1 - alpha) * right_home + alpha * right_table
                left_euler_t = (1 - alpha) * left_euler + alpha * left_euler_fwd
                right_euler_t = (1 - alpha) * right_euler + alpha * right_euler_fwd
                lg_action, rg_action = OPEN, OPEN
            elif step < 250:
                # Phase 1: hold above table, close grippers
                left_target = left_table.clone()
                right_target = right_table.clone()
                left_euler_t = left_euler_fwd
                right_euler_t = right_euler_fwd
                lg_action, rg_action = CLOSE, CLOSE
            else:
                # Phase 2: return home + restore orientation
                alpha = (step - 250) / 100.0
                left_target = (1 - alpha) * left_table + alpha * left_home
                right_target = (1 - alpha) * right_table + alpha * right_home
                left_euler_t = (1 - alpha) * left_euler_fwd + alpha * left_euler
                right_euler_t = (1 - alpha) * right_euler_fwd + alpha * right_euler
                lg_action, rg_action = OPEN, OPEN

            left_action = torch.cat([left_target, left_euler_t], dim=-1).clone()
            right_action = torch.cat([right_target, right_euler_t], dim=-1).clone()
            action = {
                "left_arm": left_action,
                "right_arm": right_action,
                "left_gripper": lg_action,
                "right_gripper": rg_action,
            }
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 50 == 0:
                left_actual = obs["tianji"]["left_arm"]["eef_position"]
                right_actual = obs["tianji"]["right_arm"]["eef_position"]
                left_err = torch.norm(left_actual - left_target).item()
                right_err = torch.norm(right_actual - right_target).item()
                print(
                    f"Episode {episode}, Step {step:3d}: "
                    f"L_err={left_err:.4f}, R_err={right_err:.4f}"
                )

    env.close()

    for episode in range(N_VIDEOS):
        step = episode * total_steps
        video_path = os.path.join(VIDEO_DIR, f"episode_{episode}_step_{step}.mp4")
        assert os.path.exists(video_path), f"Video not saved to {video_path}"
        print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    test_table_and_tianji_marvin_robotiq()
