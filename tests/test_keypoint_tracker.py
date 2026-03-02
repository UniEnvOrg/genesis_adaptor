"""Tests for KeypointTrackerNode."""

from __future__ import annotations

from typing import Optional

import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors import (
    CameraSensorNode,
    CameraSensorNodeOptions,
    KeypointTrackerNode,
    UniformSegmentationPointProposer,
)


class MovingBoxNode(WorldNode):
    """Deterministically translate a box each step."""

    reload_priorities = {20}
    reset_priorities = {20}
    pre_environment_step_priorities = {20}
    post_environment_step_priorities = {20}

    def __init__(
        self,
        world: GenesisWorld,
        name: str = "moving_box",
        *,
        start_pos=(0.0, 0.0, 0.20),
        step_delta=(0.01, 0.0, 0.0),
    ):
        self.world = world
        self._name = name
        self.start_pos = torch.tensor(start_pos, dtype=torch.float32)
        self.step_delta = torch.tensor(step_delta, dtype=torch.float32)
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
            gs.morphs.Box(size=(0.12, 0.12, 0.12), fixed=True, pos=tuple(self.start_pos.tolist())),
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


def _make_env(render_position: bool, render_depth: bool):
    world = GenesisWorld()
    moving_box = MovingBoxNode(world, step_delta=(0.012, -0.004, 0.0))

    cam_pos = (1.0, 0.0, 0.65)
    cam_lookat = (0.0, 0.0, 0.20)
    cam_opt = CameraSensorNodeOptions(
        res=(96, 96),
        pos=cam_pos,
        quat=CameraSensorNodeOptions.quat_from_lookat(cam_pos, cam_lookat),
        fov=45,
        GUI=False,
    )
    camera = CameraSensorNode(
        world=world,
        name="camera",
        option=cam_opt,
        render_rgb=False,
        render_depth=render_depth,
        render_seg=True,
        render_normal=False,
        render_position=render_position,
    )
    tracker = KeypointTrackerNode(
        world=world,
        name="gt_flow",
        camera_node=camera,
        max_points=32,
        point_proposer=UniformSegmentationPointProposer(
            max_points_per_entity=32,
            randomize=False,
            tracked_entity_names=["moving_box"],
        ),
    )
    env = WorldEnv(world, [moving_box, camera, tracker], render_mode=None)
    return env, moving_box


def _assert_tracking_translation(env: WorldEnv, moving_box: MovingBoxNode):
    _, obs, _ = env.reset(seed=0, reload=True)
    track0 = obs["gt_flow"]
    valid0 = track0["valid_masks"].bool()
    assert torch.any(valid0), "No keypoints were initialized for moving_box."

    p0 = track0["point_positions"][valid0].clone()
    delta = moving_box.step_delta

    for i in range(1, 4):
        obs, _, _, _, _ = env.step(None)
        track = obs["gt_flow"]
        valid = track["valid_masks"].bool()
        assert torch.equal(valid, valid0), "Tracked keypoint mask changed unexpectedly."

        flow = track["point_movements"][valid]
        expected_flow = delta.unsqueeze(0).expand_as(flow)
        assert torch.allclose(flow, expected_flow, atol=2e-3), (
            f"Flow mismatch at step {i}: got {flow[:3]}, expected {expected_flow[:3]}"
        )

        points = track["point_positions"][valid]
        expected_points = p0 + delta.unsqueeze(0) * i
        assert torch.allclose(points, expected_points, atol=2e-3), (
            f"Point mismatch at step {i}: got {points[:3]}, expected {expected_points[:3]}"
        )

    env.close()


def test_uniform_segmentation_point_proposer():
    proposer = UniformSegmentationPointProposer(max_points_per_entity=2, randomize=False)
    seg = torch.tensor(
        [
            [0, 1, 1, 2],
            [2, 2, 1, 0],
            [3, 1, 2, 2],
        ],
        dtype=torch.int32,
    )
    pos = torch.zeros((3, 4, 3), dtype=torch.float32)
    valid = torch.tensor(
        [
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    # Set invalid pixels to NaN so the proposer's internal isfinite check
    # matches the old valid_mask behavior.
    pos[~valid] = float("nan")

    # Propose expects batched input: (B, H, W) and (B, H, W, 3).
    seg_b = seg.unsqueeze(0)
    pos_b = pos.unsqueeze(0)

    seg_to_entity = {1: 10, 2: 11}
    entity_names = {10: "entity_a", 11: "entity_b"}
    max_pts = 3
    points_world, seg_ids, valid_mask = proposer.propose(
        camera_obs={"seg": seg_b, "position": pos_b},
        seg_to_entity_idx=seg_to_entity,
        entity_names=entity_names,
        max_points=max_pts,
    )
    assert points_world.shape == (1, max_pts, 3)
    assert seg_ids.shape == (1, max_pts)
    assert valid_mask.shape == (1, max_pts)

    n_valid = int(valid_mask[0].sum().item())
    assert n_valid <= max_pts
    assert n_valid > 0

    valid_seg = seg_ids[0, valid_mask[0]]
    for i in range(valid_seg.shape[0]):
        assert int(valid_seg[i].item()) in seg_to_entity

    for seg_id_val in torch.unique(valid_seg):
        assert int((valid_seg == seg_id_val).sum().item()) <= 2


def test_gt_keypoint_flow_sensor_with_position_map():
    env, moving_box = _make_env(render_position=True, render_depth=False)
    _assert_tracking_translation(env, moving_box)


def test_gt_keypoint_flow_sensor_with_depth_fallback():
    env, moving_box = _make_env(render_position=False, render_depth=True)
    _assert_tracking_translation(env, moving_box)
