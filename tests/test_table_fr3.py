"""Test: load a table with random texture and a Franka FR3 robot using WorldEnv."""

import os
import torch
import genesis as gs
from genesis.constants import backend as gs_backend
gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import CameraSensorNode, CameraSensorNodeOptions
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_genesis_collection.robots.franka_fr3 import FrankaFR3
from unienv_interface.world import WorldEnv
from unienv_interface.wrapper.video_record import EpisodeVideoWrapper

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos_fr3")

def test_table_and_fr3():
    """Test scene with two tables, a Franka FR3 robot, and a floor plane.

    Scene layout (top-down, X axis points right):

        y
        ^
        |   [table2]   [  empty table  ]
        |   (-1.0,0)         (0,0)
        +----------------------------> x

    Constraints:
      - The empty table is always centred at x,y = (0, 0). Do not move it.
        Its long axis runs along Y (90° Z rotation) so the robot faces across
        the narrow 0.6 m depth — giving a natural "sitting at the desk" posture.
      - The robot's mounting table (table2) and robot may be repositioned along
        X to adjust reach distance, but must remain to the left (x < -0.8) to
        avoid leg-clipping with the rotated empty table.
    """
    world = GenesisWorld()

    TABLE_SURFACE_Z = 0.6

    # Ground floor plane makes the scene look nicer
    plane = PlaneBuilder(world, name="floor", height=0.0)

    # Empty table: fixed at (0, 0), rotated 90° around Z so the long side runs
    # along Y and the robot faces across the narrow depth (0.6 m).
    # NOTE: always keep pos x,y = (0, 0).
    table = TableBuilder(world, name="table", pos=(0, 0, TABLE_SURFACE_Z), quat=(0.7071068, 0, 0, 0.7071068))
    # Robot's mounting table at negative x; inner edge at x = -0.5 (0.2 m clear of the
    # rotated empty table's left edge at x = -0.3).
    table2 = TableBuilder(world, name="table2", pos=(-1.0, 0, TABLE_SURFACE_Z))

    # Robot near the inner edge of table2 (0.2 m inside), facing +x toward the desk.
    # Identity quaternion → robot's local x axis aligns with world +x.
    franka = FrankaFR3(
        world,
        name="franka",
        pos=(-0.7, 0, TABLE_SURFACE_Z),  # near the inner edge of table2
        quat=(1, 0, 0, 0),               # identity → faces +x (toward empty table)
    )

    # Camera positioned behind the robot (negative x side), looking toward the empty table
    cam_pos = (-2.3, -1.5, 2.2)
    cam_lookat = (-0.3, 0.0, 0.6)
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

    N_VIDEOS = 3

    env = EpisodeVideoWrapper(
        WorldEnv(world, [plane, table, table2, franka, camera], render_mode="auto"),
        store_dir=VIDEO_DIR,
        format="mp4",
    )

    # Each reset rebuilds the scene with a new random texture on both tables.
    # EpisodeVideoWrapper saves one video per episode.
    for episode in range(N_VIDEOS):
        context, obs, info = env.reset(seed=episode, reload=True)

        assert obs is not None, f"Expected observation from Franka (episode {episode})"

        # Hold the initial EE pose (pos + euler = 6D) throughout the episode
        eef_pos = obs["franka"]["eef_position"]
        eef_quat = obs["franka"]["eef_quaternion"]
        rot_mat = quaternion_to_matrix(eef_quat)
        euler = matrix_to_euler_angles(rot_mat, "XYZ")
        action = torch.cat([eef_pos, euler], dim=-1)

        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None

    env.close()

    # EpisodeVideoWrapper names files episode_{n}_step_{cumulative_step}.mp4
    for episode in range(N_VIDEOS):
        step = episode * 100
        video_path = os.path.join(VIDEO_DIR, f"episode_{episode}_step_{step}.mp4")
        assert os.path.exists(video_path), f"Video not saved to {video_path}"
        print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    test_table_and_fr3()
