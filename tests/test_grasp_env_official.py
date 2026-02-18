"""Test the official Genesis grasp_env example and render a video.

Runs the grasp_and_lift_demo() from GraspEnv with 1 environment,
captures frames from the left stereo camera, and saves an MP4.
"""

import os
import math
import torch
import imageio
import numpy as np
from typing import Literal

import genesis as gs
from genesis.constants import backend as gs_backend
from genesis.utils.geom import (
    xyz_to_quat,
    transform_quat_by_quat,
    transform_by_quat,
)
from unienv_genesis_collection.asset_path import get_asset_path

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos_grasp_env_official")
os.makedirs(VIDEO_DIR, exist_ok=True)


# ── verbatim copy of GraspEnv from the official Genesis example ──────────────

class GraspEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, env_cfg["num_envs"])))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            device=gs.device,
        )

        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["box_size"],
                fixed=env_cfg["box_fixed"],
                collision=env_cfg["box_collision"],
                batch_fixed_verts=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)),
            ),
        )

        # Recording camera (wider view for the video)
        self.record_cam = self.scene.add_camera(
            res=(640, 480),
            pos=(1.5, 1.5, 1.5),
            lookat=(0.3, 0.0, 0.1),
            fov=60,
            GUI=False,
        )

        self.scene.build(n_envs=env_cfg["num_envs"])
        self.robot.set_pd_gains()

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, device=self.device, unit_length=0.5
        )
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0
        self.robot.reset(envs_idx)

        num_reset = len(envs_idx)
        random_x = torch.rand(num_reset, device=self.device) * 0.4 + 0.2
        random_y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.5
        random_z = torch.ones(num_reset, device=self.device) * 0.025
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        random_yaw = (torch.rand(num_reset, device=self.device) * 2 * math.pi - math.pi) * 0.25
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(num_reset, device=self.device),
                torch.zeros(num_reset, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)
        self.goal_pose[envs_idx] = torch.cat([random_pos, goal_yaw], dim=-1)
        self.object.set_pos(random_pos, envs_idx=envs_idx)
        self.object.set_quat(goal_yaw, envs_idx=envs_idx)

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        obs, self.extras = self.get_observations()
        return obs, self.extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        finger_pos = self.robot.center_finger_pose[:, :3]
        finger_quat = self.robot.center_finger_pose[:, 3:7]
        obj_pos = self.object.get_pos()
        obj_quat = self.object.get_quat()
        obs_components = [finger_pos - obj_pos, finger_quat, obj_pos, obj_quat]
        obs_tensor = torch.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor, self.extras

    def _reward_keypoints(self) -> torch.Tensor:
        keypoints_offset = self.keypoints_offset
        finger_tip_z_offset = torch.tensor(
            [0.0, 0.0, -0.06], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        finger_pos_keypoints = self._to_world_frame(
            self.robot.center_finger_pose[:, :3] + finger_tip_z_offset,
            self.robot.center_finger_pose[:, 3:7],
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(
            self.object.get_pos(), self.object.get_quat(), keypoints_offset
        )
        dist = torch.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    def _to_world_frame(self, position, quaternion, keypoints_offset):
        world = torch.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(keypoints_offset[:, k], quaternion)
        return world

    @staticmethod
    def get_keypoint_offsets(batch_size, device, unit_length=0.5):
        keypoint_offsets = (
            torch.tensor(
                [[0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0],
                 [0, -1.0, 0], [0, 1.0, 0], [0, 0, -1.0], [0, 0, 1.0]],
                device=device, dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets[None].repeat((batch_size, 1, 1))

    def grasp_and_lift_demo(self, frames: list | None = None, capture_every: int = 5) -> None:
        total_steps = 500
        goal_pose = self.robot.ee_pose.clone()
        lift_height = 0.3
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3
        final_pose[:, 1] = 0.0
        final_pose[:, 2] = 0.4
        reset_pose = torch.tensor(
            [0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)

        for i in range(total_steps):
            if i < total_steps / 4:
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps / 2:
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 3 / 4:
                self.robot.go_to_goal(final_pose, open_gripper=False)
            else:
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            self.scene.step()

            if frames is not None and i % capture_every == 0:
                rgb, _, _, _ = self.record_cam.render(rgb=True, depth=False, segmentation=False, normal=False)
                # rgb is a numpy array; shape (H, W, 4) or (B, H, W, 4)
                if isinstance(rgb, np.ndarray):
                    frame = rgb[0] if rgb.ndim == 4 else rgb
                else:
                    frame = (rgb[0] if rgb.ndim == 4 else rgb).cpu().numpy()
                frame_np = frame.astype(np.uint8)[..., :3]  # drop alpha
                frames.append(frame_np)
                if i % 50 == 0:
                    print(f"  step {i}/{total_steps}, frame captured")


class Manipulator:
    def __init__(self, num_envs, scene, args, device="cpu"):
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        material = gs.materials.Rigid()
        morph = gs.morphs.MJCF(
            file=get_asset_path("robots", "franka_emika_panda", "panda.xml"),
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
        self._robot_entity = scene.add_entity(material=material, morph=morph)
        self._gripper_open_dof = 0.04
        self._gripper_close_dof = 0.00
        self._ik_method: Literal["gs_ik", "dls_ik"] = args["ik_method"]
        self._init()

    def set_pd_gains(self):
        # self._robot_entity.set_dofs_kp(torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        # self._robot_entity.set_dofs_kv(torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        # self._robot_entity.set_dofs_force_range(
        #     torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        #     torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        # )
        pass

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 2
        self._gripper_dim = 2
        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(
            self._arm_dof_dim, self._arm_dof_dim + self._gripper_dim, device=self._device
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        self._default_joint_angles = list(self._args["default_arm_dof"])
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += list(self._args["default_gripper_dof"])

    def reset(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def go_to_goal(self, goal_pose, open_gripper=True):
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        self._robot_entity.control_dofs_position(
            q_pos[:, self._arm_dof_idx],
            dofs_idx_local=self._arm_dof_idx,
        )
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        # self._robot_entity.control_dofs_position(position=q_pos)
        self._robot_entity.control_dofs_position(
            q_pos[:, self._fingers_dof],
            dofs_idx_local=self._fingers_dof,
        )

    @property
    def ee_pose(self):
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self):
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self):
        pos, quat = self._right_finger_link.get_pos(), self._right_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self):
        left = self.left_finger_pose
        right = self.right_finger_pose
        center_pos = (left[:, :3] + right[:, :3]) / 2
        center_quat = left[:, 3:7]
        return torch.cat([center_pos, center_quat], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────

def test_grasp_env_official():
    gs.init(backend=gs_backend.cpu)

    env_cfg = {
        "num_envs": 1,
        "num_obs": 14,          # 3 (pos diff) + 4 (finger quat) + 3 (obj pos) + 4 (obj quat)
        "num_actions": 6,
        "image_resolution": (128, 128),
        "ctrl_dt": 0.02,
        "episode_length_s": 10.0,
        "action_scales": [0.01, 0.01, 0.01, 0.1, 0.1, 0.1],
        "use_rasterizer": True,
        "box_size": (0.05, 0.05, 0.05),
        "box_fixed": False,
        "box_collision": True,
        "visualize_camera": False,
    }
    reward_cfg = {"keypoints": 1.0}
    robot_cfg = {
        "ik_method": "gs_ik",
        "ee_link_name": "hand",
        "gripper_link_names": ["left_finger", "right_finger"],
        "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "default_gripper_dof": [0.04, 0.04],
    }

    print("Building GraspEnv...")
    env = GraspEnv(env_cfg=env_cfg, reward_cfg=reward_cfg, robot_cfg=robot_cfg, show_viewer=False)

    print("Running grasp_and_lift_demo (500 steps)...")
    frames = []
    env.grasp_and_lift_demo(frames=frames, capture_every=5)

    video_path = os.path.join(VIDEO_DIR, "grasp_env_official.mp4")
    print(f"Saving {len(frames)} frames to {video_path} ...")
    imageio.mimwrite(video_path, frames, fps=20, quality=8)
    print(f"Video saved: {video_path}")
    assert os.path.exists(video_path), "Video file not created!"


if __name__ == "__main__":
    test_grasp_env_official()
