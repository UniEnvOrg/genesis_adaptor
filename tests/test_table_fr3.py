"""Test: load a table with random texture and a Franka FR3 robot using WorldEnv."""

import os
import torch
import genesis as gs
gs.init(backend=gs.gpu)

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import CameraSensorNode, CameraSensorNodeOptions
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_genesis_collection.robots.franka_fr3 import FrankaFR3
from unienv_interface.world import WorldEnv
from unienv_interface.wrapper.video_record import EpisodeVideoWrapper

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos")


def test_table_and_fr3():
    world = GenesisWorld()

    TABLE_SURFACE_Z = 0.8
    table = TableBuilder(world, name="table", pos=(0, 0, TABLE_SURFACE_Z))
    franka = FrankaFR3(world, name="franka", pos=(0, 0, TABLE_SURFACE_Z))

    # Camera looking down at the table/robot from an angled viewpoint
    cam_pos = (1.5, -1.5, 1.8)
    cam_lookat = (0.0, 0.0, 0.8)
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

    env = EpisodeVideoWrapper(
        WorldEnv(world, [table, franka, camera], render_mode="auto"),
        store_dir=VIDEO_DIR,
        format="mp4",
    )

    context, obs, info = env.reset()

    # Franka should produce an observation; table should not
    assert obs is not None, "Expected observation from Franka"

    # Build an action that holds the current EE pose (pos + euler = 6D)
    from xbarray.transformations.rotation_conversions.pytorch import (
        quaternion_to_matrix, matrix_to_euler_angles,
    )
    eef_pos = obs["franka"]["eef_position"]
    eef_quat = obs["franka"]["eef_quaternion"]
    rot_mat = quaternion_to_matrix(eef_quat)
    euler = matrix_to_euler_angles(rot_mat, "XYZ")
    action = torch.cat([eef_pos, euler], dim=-1)

    # Run simulation steps holding the initial pose
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(action)

    assert obs is not None

    env.close()

    video_path = os.path.join(VIDEO_DIR, "episode_0_step_0.mp4")
    assert os.path.exists(video_path), f"Video not saved to {video_path}"
    print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    test_table_and_fr3()
