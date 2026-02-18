"""Minimal IK debug scene: floor plane + Franka Emika Panda only.

Runs a simple reach-down → close → lift → open cycle via the EEF controller
and saves a video so we can verify IK is tracking visually.
"""

import os
import torch
import genesis as gs
from genesis.constants import backend as gs_backend
gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors.camera_node import CameraSensorNode, CameraSensorNodeOptions
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.robots.franka_fr3_hand import FrankaFR3WithHand
from unienv_interface.world import WorldEnv
from unienv_interface.wrapper.video_record import EpisodeVideoWrapper

from xbarray.transformations.rotation_conversions.pytorch import (
    quaternion_to_matrix, matrix_to_euler_angles,
)

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos_emika_simple")


def test_emika_simple():
    world = GenesisWorld()

    plane = PlaneBuilder(world, name="floor", height=0.0)

    # Robot on the floor, facing +x
    robot = FrankaFR3WithHand(
        world,
        name="franka",
        pos=(0.0, 0.0, 0.0),
        quat=(1, 0, 0, 0),
    )

    cam_pos = (2.0, -1.5, 1.5)
    cam_lookat = (0.4, 0.0, 0.4)
    camera = CameraSensorNode(
        world,
        name="camera",
        option=CameraSensorNodeOptions(
            pos=cam_pos,
            quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
            res=(640, 480),
            fov=50,
        ),
        render_rgb=True,
    )

    env = EpisodeVideoWrapper(
        WorldEnv(world, [plane, robot, camera], render_mode="auto"),
        store_dir=VIDEO_DIR,
        format="mp4",
    )

    context, obs, info = env.reset(seed=0, reload=True)
    print("Reset done.")

    arm_obs = obs["franka"]["arm"]
    eef_pos   = arm_obs["eef_position"].clone()    # (3,)
    eef_quat  = arm_obs["eef_quaternion"].clone()  # (4,)
    rot_mat   = quaternion_to_matrix(eef_quat)
    euler     = matrix_to_euler_angles(rot_mat, "XYZ")

    print(f"Initial EEF position : {eef_pos}")
    print(f"Initial EEF euler    : {euler}")

    # Build four goal positions (all share the initial orientation)
    home_pos   = eef_pos.clone()
    lower_pos  = eef_pos.clone(); lower_pos[2] -= 0.15   # move down 15 cm
    raise_pos  = eef_pos.clone(); raise_pos[2] += 0.10   # lift 10 cm above home

    OPEN  = torch.tensor([-1.0])
    CLOSE = torch.tensor([ 1.0])

    # Simple 4-phase demo:
    #   Phase 0  (  0- 49): hold home, open gripper
    #   Phase 1  ( 50- 99): move down
    #   Phase 2  (100-149): close gripper
    #   Phase 3  (150-199): lift
    #   Phase 4  (200-249): open gripper, back to home
    total_steps = 250

    entity = robot.entity  # direct Genesis entity for low-level inspection

    for step in range(total_steps):
        if step < 50:
            target_pos = home_pos
            gripper_action = OPEN
        elif step < 100:
            alpha = (step - 50) / 50.0
            target_pos = (1 - alpha) * home_pos + alpha * lower_pos
            gripper_action = OPEN
        elif step < 150:
            target_pos = lower_pos
            gripper_action = CLOSE
        elif step < 200:
            alpha = (step - 150) / 50.0
            target_pos = (1 - alpha) * lower_pos + alpha * raise_pos
            gripper_action = CLOSE
        else:
            target_pos = raise_pos
            gripper_action = OPEN

        arm_action = torch.cat([target_pos, euler], dim=-1).clone()
        action = {"arm": arm_action, "gripper": gripper_action}

        # Snapshot joint positions BEFORE step (IK is applied in pre_environment_step)
        qpos_before = entity.get_qpos().clone()

        obs, reward, terminated, truncated, info = env.step(action)

        qpos_after = entity.get_qpos().clone()

        if step % 50 == 0:
            actual_pos = obs["franka"]["arm"]["eef_position"]
            err = torch.norm(actual_pos - target_pos).item()
            print(f"  step {step:3d}: target_z={target_pos[2]:.4f}, "
                  f"actual_z={actual_pos[2]:.4f}, err={err:.4f}")
            print(f"    qpos before: {qpos_before[:7].numpy().round(4)}")
            print(f"    qpos after : {qpos_after[:7].numpy().round(4)}")
            print(f"    qpos delta : {(qpos_after - qpos_before)[:7].numpy().round(4)}")

    env.close()

    video_path = os.path.join(VIDEO_DIR, "episode_0_step_0.mp4")
    assert os.path.exists(video_path), f"Video not saved: {video_path}"
    print(f"Video saved: {video_path}")


if __name__ == "__main__":
    test_emika_simple()
