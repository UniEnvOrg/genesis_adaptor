"""Batched tests for CameraSensorNode and KeypointTrackerNode.

These tests require CUDA (BatchRenderer) and initialize Genesis with GPU backend.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("BatchRenderer requires CUDA", allow_module_level=True)

import genesis as gs
from genesis.constants import backend as gs_backend

# Reinitialize genesis with GPU if it was already initialized (e.g. CPU).
if gs._initialized:
    gs.destroy()
gs.init(backend=gs_backend.gpu)

from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode

from unienv_genesis import GenesisWorld
from unienv_genesis.nodes.sensors import (
    CameraSensorNode,
    CameraSensorNodeOptions,
    KeypointTrackerNode,
    UniformSegmentationPointProposer,
)

N_ENVS = 2
RES = (64, 64)
CAM_POS = (1.0, 0.0, 0.5)
CAM_LOOKAT = (0.0, 0.0, 0.2)


def _cam_opt(**overrides) -> CameraSensorNodeOptions:
    kw = dict(
        res=RES,
        pos=CAM_POS,
        quat=CameraSensorNodeOptions.quat_from_lookat(CAM_POS, CAM_LOOKAT),
        fov=45,
        GUI=False,
    )
    kw.update(overrides)
    return CameraSensorNodeOptions(**kw)


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
        self.start_pos = torch.tensor(start_pos, dtype=world.dtype)
        self.step_delta = torch.tensor(step_delta, dtype=world.dtype)
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


# ------------------------------------------------------------------
# Batched camera tests
# ------------------------------------------------------------------


def _make_batched_camera_env():
    world = GenesisWorld(
        batch_size=N_ENVS,
        scene_init_kwargs={"renderer": gs.renderers.BatchRenderer(use_rasterizer=True)},
    )
    box = MovingBoxNode(world)
    camera = CameraSensorNode(
        world=world,
        name="cam",
        option=_cam_opt(),
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        render_position=True,
    )
    env = WorldEnv(world, [box, camera], render_mode=None)
    return env, camera


def test_batched_nyx_rgb_only_camera_reports_its_pose_per_env():
    """A static NYX camera without depth / seg (no secondary genesis camera)
    reports its shared pose once; the observation still carries one per env."""
    from unienv_genesis.nodes.sensors import camera_node as camera_node_module

    if camera_node_module._ensure_nyx() is not True:
        pytest.skip("NYX not available")
    world = GenesisWorld(batch_size=N_ENVS)
    box = MovingBoxNode(world)
    camera = CameraSensorNode(world=world, name="cam", option=_cam_opt(backend="nyx", spp=4), render_rgb=True)
    env = WorldEnv(world, [box, camera], render_mode=None)
    try:
        env.reset(seed=0, reload=True)
        obs = camera.get_observation()
        for key, space in camera.observation_space.spaces.items():
            assert obs[key].shape == space.shape, (key, obs[key].shape, space.shape)
        assert torch.allclose(obs["extrinsics_cv"][0], obs["extrinsics_cv"][1])
    finally:
        env.close()


def test_batched_camera_observation_shapes():
    env, camera = _make_batched_camera_env()
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()

    W, H = RES
    assert cam_obs["rgb"].shape == (N_ENVS, H, W, 3)
    assert cam_obs["depth"].shape == (N_ENVS, H, W)
    assert cam_obs["seg"].shape == (N_ENVS, H, W)
    assert cam_obs["position"].shape == (N_ENVS, H, W, 3)
    assert cam_obs["intrinsics_cv"].shape == (N_ENVS, 3, 3)
    assert cam_obs["extrinsics_cv"].shape == (N_ENVS, 3, 4)
    env.close()


def test_batched_camera_intrinsics_consistent():
    """All envs should share the same intrinsics (same camera)."""
    env, camera = _make_batched_camera_env()
    env.reset(seed=0, reload=True)
    K = camera.get_observation()["intrinsics_cv"]

    assert K.shape == (N_ENVS, 3, 3)
    for i in range(1, N_ENVS):
        assert torch.allclose(K[0], K[i])
    env.close()


def test_batched_camera_intrinsics_values():
    env, camera = _make_batched_camera_env()
    env.reset(seed=0, reload=True)
    K = camera.get_observation()["intrinsics_cv"]

    # fx == fy, bottom row = [0,0,1], skew = 0
    assert K[0, 0, 0] == K[0, 1, 1]
    assert K[0, 2, 0] == 0.0
    assert K[0, 2, 1] == 0.0
    assert K[0, 2, 2] == 1.0
    assert K[0, 0, 1] == 0.0
    assert K[0, 1, 0] == 0.0
    env.close()


def test_batched_camera_extrinsics_values():
    env, camera = _make_batched_camera_env()
    env.reset(seed=0, reload=True)
    E = camera.get_observation()["extrinsics_cv"]

    assert E.shape == (N_ENVS, 3, 4)
    for i in range(N_ENVS):
        R = E[i, :, :3]
        assert torch.allclose(R @ R.T, torch.eye(3, device=R.device), atol=1e-4)
        assert torch.allclose(torch.det(R), torch.tensor(1.0, device=R.device), atol=1e-4)
    env.close()


def test_batched_camera_observation_space_matches():
    """Observation space shape should match actual observation shapes."""
    env, camera = _make_batched_camera_env()
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()

    for key, space in camera.observation_space.spaces.items():
        assert key in cam_obs, f"Key {key!r} in obs space but not in obs"
        assert cam_obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={cam_obs[key].shape}, space={space.shape}"
        )
    env.close()


# ------------------------------------------------------------------
# Batched flow tracker tests
# ------------------------------------------------------------------


def _make_batched_flow_env(render_position: bool, render_depth: bool):
    world = GenesisWorld(
        batch_size=N_ENVS,
        scene_init_kwargs={"renderer": gs.renderers.BatchRenderer(use_rasterizer=True)},
    )
    moving_box = MovingBoxNode(world, step_delta=(0.012, -0.004, 0.0))
    camera = CameraSensorNode(
        world=world,
        name="camera",
        option=_cam_opt(res=(96, 96)),
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
    return env, moving_box, tracker


def _assert_batched_tracking(env: WorldEnv, moving_box: MovingBoxNode, tracker: KeypointTrackerNode):
    env.reset(seed=0, reload=True)
    track0 = tracker.get_observation()
    valid0 = track0["valid_masks"].bool()  # (B, max_points)
    assert valid0.ndim == 2 and valid0.shape[0] == N_ENVS
    assert torch.any(valid0), "No keypoints were initialized."

    p0 = track0["point_positions"].clone()  # (B, max_points, 3)
    delta = moving_box.step_delta

    for i in range(1, 4):
        env.step(None)
        track = tracker.get_observation()
        valid = track["valid_masks"].bool()
        assert torch.equal(valid, valid0), "Tracked keypoint mask changed."

        for b in range(N_ENVS):
            v = valid[b]
            if not torch.any(v):
                continue
            flow = track["point_movements"][b][v]
            expected_flow = delta.to(flow.device).unsqueeze(0).expand_as(flow)
            assert torch.allclose(flow, expected_flow, atol=2e-3), (
                f"Flow mismatch at step {i}, env {b}"
            )
            points = track["point_positions"][b][v]
            expected = p0[b][v] + delta.to(points.device).unsqueeze(0) * i
            assert torch.allclose(points, expected, atol=2e-3), (
                f"Point mismatch at step {i}, env {b}"
            )

    env.close()


def test_batched_flow_with_position_map():
    env, moving_box, tracker = _make_batched_flow_env(render_position=True, render_depth=False)
    _assert_batched_tracking(env, moving_box, tracker)


def test_batched_flow_with_depth_fallback():
    env, moving_box, tracker = _make_batched_flow_env(render_position=False, render_depth=True)
    _assert_batched_tracking(env, moving_box, tracker)
