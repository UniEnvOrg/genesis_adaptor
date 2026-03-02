"""Tests for CameraSensorNode observation shapes and intrinsics/extrinsics (unbatched, CPU)."""

from __future__ import annotations

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
)


class DummyBoxNode(WorldNode):
    """Static box entity so the scene is non-empty."""

    reload_priorities = {20}
    reset_priorities = {20}
    pre_environment_step_priorities = {20}
    post_environment_step_priorities = {20}

    def __init__(self, world: GenesisWorld, name: str = "box"):
        self.world = world
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reload(self, *, priority=0, **kw):
        if priority == 20:
            self.world.scene.add_entity(
                gs.morphs.Box(size=(0.1, 0.1, 0.1), fixed=True, pos=(0, 0, 0.2)),
                name=self._name,
            )

    def reset(self, *, priority=0, **kw):
        pass

    def pre_environment_step(self, dt, *, priority=0):
        pass

    def post_environment_step(self, dt, *, priority=0):
        pass

    def get_observation(self):
        return None


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


def _make_unbatched_env():
    world = GenesisWorld()
    box = DummyBoxNode(world)
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


def test_unbatched_observation_shapes():
    env, camera = _make_unbatched_env()
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()

    W, H = RES
    assert cam_obs["rgb"].shape == (H, W, 3)
    assert cam_obs["depth"].shape == (H, W)
    assert cam_obs["seg"].shape == (H, W)
    assert cam_obs["position"].shape == (H, W, 3)
    assert cam_obs["intrinsics_cv"].shape == (3, 3)
    assert cam_obs["extrinsics_cv"].shape == (3, 4)
    env.close()


def test_unbatched_intrinsics_values():
    env, camera = _make_unbatched_env()
    env.reset(seed=0, reload=True)
    K = camera.get_observation()["intrinsics_cv"]

    assert K.shape == (3, 3)
    # fx == fy (single focal length in Genesis)
    assert K[0, 0] == K[1, 1]
    # Bottom row is [0, 0, 1]
    assert K[2, 0] == 0.0
    assert K[2, 1] == 0.0
    assert K[2, 2] == 1.0
    # Skew is zero
    assert K[0, 1] == 0.0
    assert K[1, 0] == 0.0
    env.close()


def test_unbatched_extrinsics_values():
    env, camera = _make_unbatched_env()
    env.reset(seed=0, reload=True)
    E = camera.get_observation()["extrinsics_cv"]

    assert E.shape == (3, 4)
    R = E[:, :3]
    # R should be a valid rotation: R @ R^T ~= I, det(R) ~= 1
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-4)
    env.close()


def test_unbatched_observation_space_matches():
    """Observation space shape should match actual observation shapes."""
    env, camera = _make_unbatched_env()
    env.reset(seed=0, reload=True)
    cam_obs = camera.get_observation()

    for key, space in camera.observation_space.spaces.items():
        assert key in cam_obs, f"Key {key!r} in obs space but not in obs"
        assert cam_obs[key].shape == space.shape, (
            f"Shape mismatch for {key!r}: obs={cam_obs[key].shape}, space={space.shape}"
        )
    env.close()
