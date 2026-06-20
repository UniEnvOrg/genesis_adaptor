"""Example: keypoint tracking with a video output.

Sets up a Genesis scene with a floor plane and three moving boxes of
different sizes, a static camera, and a ``KeypointTrackerNode`` that
proposes and tracks points on every visible rigid entity. The per-step
camera RGB frames are overlaid with the tracked keypoints (green when
visible, red when occluded/out-of-frame) and written to an MP4 video.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu, logging_level="error")

import cv2

from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode
from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors import (
    CameraSensorNode,
    CameraSensorNodeOptions,
    KeypointTrackerNode,
    UniformSegmentationPointProposer,
)
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder


VIDEO_DIR = os.path.join(os.path.dirname(__file__), "videos_keypoint_tracking")
TOTAL_STEPS = 60
FPS = 15


class MovingBoxNode(WorldNode):
    """Deterministically translate a fixed box each step."""

    reload_priorities = {20}
    reset_priorities = {20}
    pre_environment_step_priorities = {20}
    post_environment_step_priorities = {20}

    def __init__(
        self,
        world: GenesisWorld,
        name: str,
        *,
        start_pos=(0.0, 0.0, 0.20),
        step_delta=(0.01, 0.0, 0.0),
        size=(0.12, 0.12, 0.12),
        color=None,
    ):
        # ``color`` is accepted for API completeness but not applied: Genesis
        # assigns default visual materials, and per-box colors are not needed
        # for this demo (boxes are distinguished by size and motion).
        del color

        self.world = world
        self._name = name
        self.start_pos = torch.tensor(start_pos, dtype=world.dtype)
        self.step_delta = torch.tensor(step_delta, dtype=world.dtype)
        self.size = tuple(size)
        self._step = 0
        self.entity = None

    @property
    def name(self) -> str:
        return self._name

    def reload(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        del seed, mask, kwargs
        if priority != 20:
            return
        self.entity = self.world.scene.add_entity(
            gs.morphs.Box(size=self.size, fixed=True, pos=tuple(self.start_pos.tolist())),
            name=self._name,
        )

    def reset(self, *, priority: int = 0, seed: Optional[int] = None, mask=None, **kwargs):
        del seed, mask, kwargs
        if priority != 20:
            return
        self._step = 0
        self.entity.set_pos(self.start_pos)

    def post_environment_step(self, dt, *, priority: int = 0):
        del dt
        if priority != 20:
            return
        self._step += 1
        self.entity.set_pos(self.start_pos + self.step_delta * self._step)

    def pre_environment_step(self, dt, *, priority: int = 0):
        del dt, priority

    def get_observation(self):
        return None


def _rgb_to_numpy_uint8(rgb_tensor: torch.Tensor) -> np.ndarray:
    """Convert a camera RGB tensor to a ``(H, W, 3)`` uint8 numpy array."""
    if rgb_tensor.dtype != torch.uint8:
        rgb_tensor = rgb_tensor.clamp(0, 255).to(torch.uint8)
    return rgb_tensor.detach().cpu().numpy()


def _overlay_keypoints(
    frame_bgr: np.ndarray,
    points_2d: torch.Tensor,
    valid_masks: torch.Tensor,
    visibility_masks: torch.Tensor,
    step: int,
) -> np.ndarray:
    """Draw tracked keypoints and a text overlay onto a BGR frame (in place)."""
    valid = valid_masks.bool()
    visible = visibility_masks.bool()
    n_valid = int(valid.sum().item())
    n_visible = int((valid & visible).sum().item())

    pts = points_2d.detach().cpu().numpy()
    for idx in np.nonzero(valid.cpu().numpy())[0]:
        u, v = pts[idx]
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        center = (int(round(float(u))), int(round(float(v))))
        if visible[idx].item():
            cv2.circle(frame_bgr, center, 4, (0, 255, 0), -1)
        else:
            cv2.circle(frame_bgr, center, 3, (0, 0, 255), -1)

    cv2.putText(
        frame_bgr,
        f"Step {step}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        f"Valid: {n_valid}  Visible: {n_visible}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame_bgr


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # -- Scene --
    vis_options = gs.options.VisOptions(segmentation_level="link")
    world = GenesisWorld(scene_init_kwargs={"vis_options": vis_options})

    plane = PlaneBuilder(world, name="floor", height=0.0)

    boxes = [
        MovingBoxNode(
            world, name="red_box",
            start_pos=(0.0, 0.0, 0.10),
            step_delta=(0.008, 0.0, 0.0),
            size=(0.12, 0.12, 0.12),
        ),
        MovingBoxNode(
            world, name="green_box",
            start_pos=(0.3, 0.3, 0.08),
            step_delta=(-0.005, -0.006, 0.0),
            size=(0.10, 0.10, 0.10),
        ),
        MovingBoxNode(
            world, name="blue_box",
            start_pos=(-0.3, 0.2, 0.06),
            step_delta=(0.004, -0.007, 0.0),
            size=(0.08, 0.08, 0.08),
        ),
    ]

    # -- Camera --
    cam_pos = (1.5, 0.0, 1.0)
    cam_lookat = (0.0, 0.0, 0.2)
    cam_opt = CameraSensorNodeOptions(
        res=(480, 480),
        pos=cam_pos,
        quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
        fov=45,
        GUI=False,
    )
    camera = CameraSensorNode(
        world=world,
        name="camera",
        option=cam_opt,
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        render_normal=False,
        render_position=True,
    )

    # -- Keypoint tracker (no entity filter -> tracks all boxes) --
    tracker = KeypointTrackerNode(
        world=world,
        name="keypoint_tracker",
        camera_node=camera,
        max_points=64,
        point_proposer=UniformSegmentationPointProposer(
            max_points_per_entity=20,
            randomize=False,
        ),
    )

    # -- Environment --
    env = WorldEnv(
        world,
        [plane, boxes[0], boxes[1], boxes[2], camera, tracker],
        render_mode=None,
    )

    # -- Episode --
    frames = []

    def capture(step: int):
        rgb = camera.get_observation()["rgb"]
        frame_rgb = _rgb_to_numpy_uint8(rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        track = tracker.get_observation()
        _overlay_keypoints(
            frame_bgr,
            track["points_2d"],
            track["valid_masks"],
            track["visibility_masks"],
            step,
        )
        frames.append(frame_bgr)
        n_valid = int(track["valid_masks"].bool().sum().item())
        n_visible = int(
            (track["valid_masks"].bool() & track["visibility_masks"].bool()).sum().item()
        )
        print(f"Step {step}: valid={n_valid} visible={n_visible}")

    env.reset(seed=0, reload=True)
    capture(0)

    for step in range(1, TOTAL_STEPS + 1):
        env.step(None)
        capture(step)

    # -- Write video --
    H, W = frames[0].shape[:2]
    video_path = os.path.join(VIDEO_DIR, "keypoint_tracking_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, float(FPS), (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()

    env.close()

    size_bytes = os.path.getsize(video_path)
    print(f"Saved video to {video_path} ({size_bytes / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
