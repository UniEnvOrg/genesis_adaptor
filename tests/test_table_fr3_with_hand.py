"""Test: load a table with random texture and a Franka FR3 + Hand robot using WorldEnv.

Gripper control verification
-----------------------------
The combined controller exposes a nested action/observation structure:

    action = {
        "arm":     <6-D eef_pose tensor>   (pos[3] + euler[3])
        "gripper": <1-D scalar tensor>      (-1 = open, +1 = close)
    }

    obs["franka"]["arm"]["eef_position"]         (3,)
    obs["franka"]["arm"]["eef_quaternion"]        (4,)
    obs["franka"]["gripper"]["joint_position"]    (2,)  # both finger joints
"""

import os
import torch
import genesis as gs
from genesis.constants import backend as gs_backend
gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import CameraSensorNode, CameraSensorNodeOptions
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.table_builder import TableBuilder
from unienv_genesis_collection.robots.franka_fr3_hand import FrankaFR3WithHand
from unienv_interface.world import WorldEnv
from unienv_interface.wrapper.video_record import EpisodeVideoWrapper

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos_fr3_with_hand")


def test_table_and_fr3_with_hand():
    """Test scene with two tables, a Franka FR3 + Hand robot, and a floor plane.

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
    franka = FrankaFR3WithHand(
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

        # --- Get initial EE pose for movement planning ---
        arm_obs = obs["franka"]["arm"]
        eef_pos = arm_obs["eef_position"].clone()
        eef_quat = arm_obs["eef_quaternion"].clone()
        rot_mat = quaternion_to_matrix(eef_quat)
        euler = matrix_to_euler_angles(rot_mat, "XYZ")

        # Define movement targets (relative to initial position)
        home_pos = eef_pos.clone()
        # Reach forward toward the empty table
        # FR3+Hand has a shorter effective reach than the Panda, so we use a
        # more conservative 0.15 m step to stay comfortably inside the workspace.
        forward_pos = home_pos.clone()
        forward_pos[0] += 0.15  # Move +x (toward table)
        # Lower position — stop 22 cm above the table surface (vs 15 cm for the
        # Panda) to avoid the joint-limit boundary at full forward extension.
        lower_pos = forward_pos.clone()
        lower_pos[2] = TABLE_SURFACE_Z + 0.22  # 22cm above table
        # Lift position (above table)
        lift_pos = lower_pos.clone()
        lift_pos[2] += 0.15
        # Side position (move to the side)
        side_pos = home_pos.clone()
        side_pos[1] += 0.15  # Move +y (to the side)

        OPEN = torch.tensor([-1.0])
        CLOSE = torch.tensor([1.0])

        # 5-phase movement demonstration:
        #   Phase 0 (  0- 49): Move forward to table, open gripper
        #   Phase 1 ( 50- 99): Lower to table surface, close gripper
        #   Phase 2 (100-149): Lift up from table, keep gripper closed
        #   Phase 3 (150-199): Move to side position, open gripper
        #   Phase 4 (200-249): Return to home position
        total_steps = 250

        for step in range(total_steps):
            if step < 50:
                # Phase 0: Move forward
                alpha = step / 50.0
                target_pos = (1 - alpha) * home_pos + alpha * forward_pos
                gripper_action = OPEN
            elif step < 100:
                # Phase 1: Lower down
                alpha = (step - 50) / 50.0
                target_pos = (1 - alpha) * forward_pos + alpha * lower_pos
                gripper_action = CLOSE
            elif step < 150:
                # Phase 2: Lift up
                alpha = (step - 100) / 50.0
                target_pos = (1 - alpha) * lower_pos + alpha * lift_pos
                gripper_action = CLOSE
            elif step < 200:
                # Phase 3: Move to side
                alpha = (step - 150) / 50.0
                target_pos = (1 - alpha) * lift_pos + alpha * side_pos
                gripper_action = OPEN
            else:
                # Phase 4: Return home
                alpha = (step - 200) / 50.0
                target_pos = (1 - alpha) * side_pos + alpha * home_pos
                gripper_action = OPEN

            arm_action = torch.cat([target_pos, euler], dim=-1).clone()
            action = {"arm": arm_action, "gripper": gripper_action}
            obs, reward, terminated, truncated, info = env.step(action)

            # Log progress at key points
            if step % 50 == 0:
                arm_obs = obs["franka"]["arm"]
                actual_pos = arm_obs["eef_position"]
                err = torch.norm(actual_pos - target_pos).item()
                print(f"Episode {episode}, Step {step:3d}: target=({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}), "
                      f"actual=({actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}), err={err:.4f}")

    env.close()

    # EpisodeVideoWrapper names files episode_{n}_step_{cumulative_step}.mp4
    for episode in range(N_VIDEOS):
        step = episode * total_steps
        video_path = os.path.join(VIDEO_DIR, f"episode_{episode}_step_{step}.mp4")
        assert os.path.exists(video_path), f"Video not saved to {video_path}"
        print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    test_table_and_fr3_with_hand()
