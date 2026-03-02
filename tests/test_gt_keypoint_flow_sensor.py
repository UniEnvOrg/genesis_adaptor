"""Tests for GTKeypointFlowSensorNode."""

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
    GTKeypointFlowSensorNode,
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
    tracker = GTKeypointFlowSensorNode(
        world=world,
        name="gt_flow",
        camera_node=camera,
        max_points=32,
        tracked_entity_names=["moving_box"],
        point_proposer=UniformSegmentationPointProposer(
            max_points_per_entity=32,
            randomize=False,
        ),
    )
    env = WorldEnv(world, [moving_box, camera, tracker], render_mode=None)
    return env, moving_box


def _assert_tracking_translation(env: WorldEnv, moving_box: MovingBoxNode):
    _, obs, _ = env.reset(seed=0, reload=True)
    track0 = obs["gt_flow"]
    valid0 = track0["valid_mask"].bool()
    assert torch.any(valid0), "No keypoints were initialized for moving_box."

    p0 = track0["points_world"][valid0].clone()
    delta = moving_box.step_delta

    for i in range(1, 4):
        obs, _, _, _, _ = env.step(None)
        track = obs["gt_flow"]
        valid = track["valid_mask"].bool()
        assert torch.equal(valid, valid0), "Tracked keypoint mask changed unexpectedly."

        flow = track["flow_world"][valid]
        expected_flow = delta.unsqueeze(0).expand_as(flow)
        assert torch.allclose(flow, expected_flow, atol=2e-3), (
            f"Flow mismatch at step {i}: got {flow[:3]}, expected {expected_flow[:3]}"
        )

        points = track["points_world"][valid]
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
    seg_to_entity = {1: 10, 2: 11}
    selected = proposer.propose(
        segmentation=seg,
        position=pos,
        valid_mask=valid,
        seg_to_entity_idx=seg_to_entity,
        max_points=3,
    )
    assert selected.shape[1] == 2
    assert selected.shape[0] <= 3
    for yx in selected:
        y, x = int(yx[0]), int(yx[1])
        assert bool(valid[y, x].item())
        assert int(seg[y, x].item()) in seg_to_entity

    seg_values = seg[selected[:, 0], selected[:, 1]]
    for seg_id in torch.unique(seg_values):
        assert int((seg_values == seg_id).sum().item()) <= 2


def test_gt_keypoint_flow_sensor_with_position_map():
    env, moving_box = _make_env(render_position=True, render_depth=False)
    _assert_tracking_translation(env, moving_box)


def test_gt_keypoint_flow_sensor_with_depth_fallback():
    env, moving_box = _make_env(render_position=False, render_depth=True)
    _assert_tracking_translation(env, moving_box)
