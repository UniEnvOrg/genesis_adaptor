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
    total_steps = 250

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

        # --- Get initial EE pose for movement planning ---
        eef_pos = obs["franka"]["eef_position"].clone()
        eef_quat = obs["franka"]["eef_quaternion"].clone()
        rot_mat = quaternion_to_matrix(eef_quat)
        euler = matrix_to_euler_angles(rot_mat, "XYZ")

        # Define movement targets (relative to initial position)
        home_pos = eef_pos.clone()
        # Reach forward toward the empty table
        forward_pos = home_pos.clone()
        forward_pos[0] += 0.3  # Move +x (toward table)
        # Lower position (down to table surface)
        lower_pos = forward_pos.clone()
        lower_pos[2] = TABLE_SURFACE_Z + 0.15  # 15cm above table
        # Lift position (above table)
        lift_pos = lower_pos.clone()
        lift_pos[2] += 0.15
        # Side position (move to the side)
        side_pos = home_pos.clone()
        side_pos[1] += 0.2  # Move +y (to the side)
        # High position (reach up)
        high_pos = home_pos.clone()
        high_pos[2] += 0.2  # Move up

        # 5-phase movement demonstration:
        #   Phase 0 (  0- 49): Move forward to table
        #   Phase 1 ( 50- 99): Lower to table surface
        #   Phase 2 (100-149): Lift up from table
        #   Phase 3 (150-199): Move to side position
        #   Phase 4 (200-249): Return to home position
        for step in range(total_steps):
            if step < 50:
                # Phase 0: Move forward
                alpha = step / 50.0
                target_pos = (1 - alpha) * home_pos + alpha * forward_pos
            elif step < 100:
                # Phase 1: Lower down
                alpha = (step - 50) / 50.0
                target_pos = (1 - alpha) * forward_pos + alpha * lower_pos
            elif step < 150:
                # Phase 2: Lift up
                alpha = (step - 100) / 50.0
                target_pos = (1 - alpha) * lower_pos + alpha * lift_pos
            elif step < 200:
                # Phase 3: Move to side
                alpha = (step - 150) / 50.0
                target_pos = (1 - alpha) * lift_pos + alpha * side_pos
            else:
                # Phase 4: Return home
                alpha = (step - 200) / 50.0
                target_pos = (1 - alpha) * side_pos + alpha * home_pos

            action = torch.cat([target_pos, euler], dim=-1).clone()
            obs, reward, terminated, truncated, info = env.step(action)

            # Log progress at key points
            if step % 50 == 0:
                actual_pos = obs["franka"]["eef_position"]
                err = torch.norm(actual_pos - target_pos).item()
                print(f"Episode {episode}, Step {step:3d}: target=({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}), "
                      f"actual=({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}), err={err:.4f}")

        assert obs is not None

    env.close()

    # EpisodeVideoWrapper names files episode_{n}_step_{cumulative_step}.mp4
    for episode in range(N_VIDEOS):
        step = episode * total_steps
        video_path = os.path.join(VIDEO_DIR, f"episode_{episode}_step_{step}.mp4")
        assert os.path.exists(video_path), f"Video not saved to {video_path}"
        print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    test_table_and_fr3()
